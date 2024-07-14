"""
Solves linear optimal power flow for a network iteratively while updating reactances.
Relevant Settings
-----------------
.. code:: yaml
    solving:
        tmpdir:
        options:
            formulation:
            clip_p_max_pu:
            load_shedding:
            noisy_costs:
            nhours:
            min_iterations:
            max_iterations:
            skip_iterations:
            track_iterations:
        solver:
            name:
.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity_cf`, :ref:`solving_cf`, :ref:`plotting_cf`
Inputs
------
- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`prepare`
Outputs
-------
- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Solved PyPSA network including optimisation results
    .. image:: ../img/results.png
        :scale: 40 %
Description
-----------
Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.
The optimization is based on the ``pyomo=False`` setting in the :func:`network.lopf` and  :func:`pypsa.linopf.ilopf` function.
Additionally, some extra constraints specified in :mod:`prepare_network` are added.
Solving the network in multiple iterations is motivated through the dependence of transmission line capacities and impedances on values of corresponding flows.
As lines are expanded their electrical parameters change, which renders the optimisation bilinear even if the power flow
equations are linearized.
To retain the computational advantage of continuous linear programming, a sequential linear programming technique
is used, where in between iterations the line impedances are updated.
Details (and errors made through this heuristic) are discussed in the paper
- Fabian Neumann and Tom Brown. `Heuristics for Transmission Expansion Planning in Low-Carbon Energy System Models <https://arxiv.org/abs/1907.10548>`_), *16th International Conference on the European Energy Market*, 2019. `arXiv:1907.10548 <https://arxiv.org/abs/1907.10548>`_.
.. warning::
    Capital costs of existing network components are not included in the objective function,
    since for the optimisation problem they are just a constant term (no influence on optimal result).
    Therefore, these capital costs are not included in ``network.objective``!
    If you want to calculate the full total annual system costs add these to the objective value.
.. tip::
    The rule :mod:`solve_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`solve_network`.
"""
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, clean_pu_profiles
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.linopf import (
    define_constraints,
    define_variables,
    get_var,
    ilopf,
    join_exprs,
    linexpr,
    network_lopf,
    get_con,
    run_and_read_cbc,
    run_and_read_cplex,
    run_and_read_glpk,
    run_and_read_gurobi,
    run_and_read_highs,
    run_and_read_xpress,
    set_conref,
    write_bound,
    write_constraint,
    write_objective,
)

from pypsa.descriptors import (
    Dict,
    additional_linkports,
    expand_series,
    get_active_assets,
    get_activity_mask,
    get_bounds_pu,
    get_extendable_i,
    get_non_extendable_i,
    nominal_attrs,
)
idx = pd.IndexSlice

from vresutils.benchmark import memory_logger

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Comment out for debugging and development

logger = logging.getLogger(__name__)


def prepare_network(n, solve_opts):
    if "clip_p_max_pu" in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    clean_pu_profiles(n)
    load_shedding = solve_opts.get("load_shedding")
    if load_shedding:
        n.add("Carrier", "Load")
        buses_i = n.buses.query("carrier == 'AC'").index
        if not np.isscalar(load_shedding):
            load_shedding = 1.0e5  # ZAR/MWh
        # intersect between macroeconomic and surveybased
        # willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full)
        # 1e2 is practical relevant, 8e3 good for debugging
        n.madd(
            "Generator",
            buses_i,
            " load_shedding",
            bus=buses_i,
            carrier="load_shedding",
            build_year=n.investment_periods[0],
            lifetime=100,
            #sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=1e5, #load_shedding, 
            p_nom=1e6,  # MW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components(n.one_port_components):
            # TODO: uncomment out to and test noisy_cost (makes solution unique)
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    return n


def add_CCL_constraints(n, sns, config):
    agg_p_nom_limits = config["electricity"].get("agg_p_nom_limits")

    try:
        agg_p_nom_minmax = pd.read_csv(agg_p_nom_limits, index_col=list(range(2)))
    except IOError:
        logger.exception(
            "Need to specify the path to a .csv file containing "
            "aggregate capacity limits per country in "
            "config['electricity']['agg_p_nom_limit']."
        )
    logger.info(
        "Adding per carrier generation capacity constraints for " "individual countries"
    )

    gen_country = n.generators.bus.map(n.buses.country)
    # cc means country and carrier
    p_nom_per_cc = (
        pd.DataFrame(
            {
                "p_nom": linexpr((1, get_var(n, "Generator", "p_nom"))),
                "country": gen_country,
                "carrier": n.generators.carrier,
            }
        )
        .dropna(subset=["p_nom"])
        .groupby(["country", "carrier"])
        .p_nom.apply(join_exprs)
    )
    minimum = agg_p_nom_minmax["min"].dropna()
    if not minimum.empty:
        minconstraint = define_constraints(
            n, p_nom_per_cc[minimum.index], ">=", minimum, "agg_p_nom", "min"
        )
    maximum = agg_p_nom_minmax["max"].dropna()
    if not maximum.empty:
        maxconstraint = define_constraints(
            n, p_nom_per_cc[maximum.index], "<=", maximum, "agg_p_nom", "max"
        )


def add_EQ_constraints(n, sns, o, scaling=1e-1):
    float_regex = "[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    lhs_gen = (
        linexpr(
            (n.snapshot_weightings.generators * scaling, get_var(n, "Generator", "p").T)
        )
        .T.groupby(ggrouper, axis=1)
        .apply(join_exprs)
    )
    lhs_spill = (
        linexpr(
            (
                -n.snapshot_weightings.stores * scaling,
                get_var(n, "StorageUnit", "spill").T,
            )
        )
        .T.groupby(sgrouper, axis=1)
        .apply(join_exprs)
    )
    lhs_spill = lhs_spill.reindex(lhs_gen.index).fillna("")
    lhs = lhs_gen + lhs_spill
    define_constraints(n, lhs, ">=", rhs, "equity", "min")

def min_capacity_factor(n,sns):
    for y in n.snapshots.get_level_values(0).unique():
        for carrier in snakemake.config["electricity"]["min_capacity_factor"]:
            # only apply to extendable generators for now
            cf = snakemake.config["electricity"]["min_capacity_factor"][carrier]
            for tech in n.generators[(n.generators.p_nom_extendable==True) & (n.generators.carrier==carrier)].index:
                tech_p_nom=get_var(n, 'Generator', 'p_nom')[tech]
                tech_p_nom=get_var(n, 'Generator', 'p_nom')[tech]
                tech_p=get_var(n, 'Generator', 'p')[tech].loc[y]
                lhs = linexpr((1,tech_p)).sum()+linexpr((-cf*8760,tech_p_nom))
                define_constraints(n, lhs, '>=',0, 'Generators', tech+'_y_'+str(y)+'_min_CF')

# Reserve requirement of 1GW for spinning acting reserves from PHS or battery, and 2.2GW of total reserves
def reserves(n, sns):

    # Operating reserves
    model_setup = pd.read_excel(
            snakemake.input.model_file,
            sheet_name='model_setup',
            index_col=[0]
    ).loc[snakemake.wildcards.model_file]

    reserve_requirements =pd.read_excel(
        snakemake.input.model_file, sheet_name='projected_parameters', index_col=[0,1]
    )

    for reserve_type in ['spinning','total']:
        carriers = snakemake.config["electricity"]["operating_reserves"][reserve_type]
        for y in n.snapshots.get_level_values(0).unique():
            lhs=0
            rhs = reserve_requirements.loc[(model_setup['projected_parameters'],reserve_type+'_reserves'),y]

            # Generators
            for tech_type in ['Generator','StorageUnit']:
                active = get_active_assets(n,tech_type,y)
                tech_list = n.df(tech_type).query("carrier == @carriers").index.intersection(active[active].index)
                for tech in tech_list:
                    if tech_type=='Generator':
                        tech_p=get_var(n, tech_type, 'p')[tech].loc[y]
                    elif tech_type=='StorageUnit':
                        tech_p=get_var(n, tech_type, 'p_dispatch')[tech].loc[y]
                    p_max_pu = get_as_dense(n, tech_type, "p_max_pu")[tech].loc[y]
                    if type(lhs)==int:
                        lhs=linexpr((-1,tech_p))
                    else:
                        lhs+=linexpr((-1,tech_p))
                    if n.df(tech_type).p_nom_extendable[tech]==False:
                        tech_p_nom=n.df(tech_type).p_nom[tech]
                        rhs+=-tech_p_nom*p_max_pu
                    else:
                        tech_p_nom=get_var(n, tech_type, 'p_nom')[tech]
                        lhs+=linexpr((p_max_pu,tech_p_nom))
            lhs.index=pd.MultiIndex.from_arrays([lhs.index.year,lhs.index])
            rhs.index=pd.MultiIndex.from_arrays([rhs.index.year,rhs.index])
            define_constraints(n, lhs, '>=',rhs, 'Reserves_'+str(y)+'_'+reserve_type)

    ###################################################################################
    # Reserve margin above maximum peak demand in each year
    # The sum of res_margin_carriers multiplied by their assumed constribution factors
    # must be higher than the maximum peak demand in each year by the reserve_margin value

    peakdemand = n.loads_t.p_set.sum(axis=1).groupby(n.snapshots.get_level_values(0)).max()
    res_margin_carriers = snakemake.config['electricity']['reserve_margin']

    for y in n.snapshots.get_level_values(0).unique():
        if reserve_requirements.loc[(model_setup['projected_parameters'],'reserve_margin_active'),y]:
            active = (
                n.generators.index[n.get_active_assets('Generator',y)]
                .append(n.storage_units.index[n.get_active_assets('StorageUnit',y)])
            ).to_list()

            exist_capacity=0
            for c in ['Generator','StorageUnit']:
                non_ext_gen_i = n.df(c).index[
                    (n.df(c).carrier.isin(res_margin_carriers)) &
                    (n.df(c).p_nom_extendable==False) &
                    (n.df(c).index.isin(active))
                ]
                exist_capacity += (
                    n.df(c).loc[non_ext_gen_i,'p_nom']
                    .mul(n.df(c).loc[non_ext_gen_i,'carrier'].map(res_margin_carriers))
                ).sum()

                ext_gen_i = n.df(c).index[
                    (n.df(c).carrier.isin(res_margin_carriers)) &
                    (n.df(c).p_nom_extendable==True) &
                    (n.df(c).index.isin(active))
                ]
                if c =='Generator':
                    lhs = linexpr(
                        (
                            n.df(c).loc[ext_gen_i,'carrier']
                            .map(res_margin_carriers),
                            get_var(n, c, "p_nom")[ext_gen_i]
                        )
                    ).sum()
                else:
                    lhs += linexpr(
                        (
                            n.df(c).loc[ext_gen_i,'carrier']
                            .map(res_margin_carriers),
                            get_var(n, c, "p_nom")[ext_gen_i]
                        )
                    ).sum()

            rhs = (peakdemand.loc[y]*(1+
                reserve_requirements.loc[(model_setup['projected_parameters'],'reserve_margin'),y])
                - exist_capacity
            )
            define_constraints(n, lhs, ">=", rhs, "reserve_margin", str(y))

def define_storage_global_constraints(n, sns):
    """
    Defines global constraints for the optimization. Possible types are.
    4. tech_capacity_expansion_limit - linopf only considers generation - so add in storage
        Use this to se a limit for the summed capacitiy of a carrier (e.g.
        'onwind') for each investment period at choosen nodes. This limit
        could e.g. represent land resource/ building restrictions for a
        technology in a certain region. Currently, only the
        capacities of extendable generators have to be below the set limit.
    """

    if n._multi_invest:
        period_weighting = n.investment_period_weightings["years"]
        weightings = n.snapshot_weightings.mul(period_weighting, level=0, axis=0).loc[
            sns
        ]
    else:
        weightings = n.snapshot_weightings.loc[sns]

    def get_period(n, glc, sns):
        period = slice(None)
        if n._multi_invest and not np.isnan(glc["investment_period"]):
            period = int(glc["investment_period"])
            if period not in sns.unique("period"):
                logger.warning(
                    "Optimized snapshots do not contain the investment "
                    f"period required for global constraint `{glc.name}`."
                )
        return period


    # (4) tech_capacity_expansion_limit
    # TODO: Generalize to carrier capacity expansion limit (i.e. also for stores etc.)
    #substr = lambda s: re.sub(r"[\[\]\(\)]", "", s)
    glcs = n.global_constraints.query("type == " '"tech_capacity_expansion_limit"')
    c, attr = "StorageUnit", "p_nom"

    for name, glc in glcs.iterrows():
        period = get_period(n, glc, sns)
        car = glc["carrier_attribute"]
        bus = str(glc.get("bus", ""))  # in pypsa buses are always strings
        ext_i = n.df(c).query("carrier == @car and p_nom_extendable").index
        if bus:
            ext_i = n.df(c).loc[ext_i].query("bus == @bus").index
        ext_i = ext_i[get_activity_mask(n, c, sns)[ext_i].loc[period].any()]

        if ext_i.empty:
            continue

        cap_vars = get_var(n, c, attr)[ext_i]

        lhs = join_exprs(linexpr((1, cap_vars)))
        rhs = glc.constant
        sense = glc.sense

        define_constraints(
            n,
            lhs,
            sense,
            rhs,
            "GlobalConstraint",
            "mu",
            axes=pd.Index([name]),
            spec=name,
        )


def add_local_max_capacity_constraint(n,snapshots):

    c, attr = 'Generator', 'p_nom'
    res = ['onwind', 'solar']
    ext_i = n.df(c)[(n.df(c)["carrier"].isin(res))
                    & (n.df(c)["p_nom_extendable"])].index
    time_valid = snapshots.levels[0]

    active_i = pd.concat([get_active_assets(n,c,inv_p,snapshots).rename(inv_p)
                          for inv_p in time_valid], axis=1).astype(int)

    ext_and_active = active_i.T[active_i.index.intersection(ext_i)]

    if ext_and_active.empty: return

    cap_vars = get_var(n, c, attr)[ext_and_active.columns]

    lhs = (linexpr((ext_and_active, cap_vars)).T
           .groupby([n.df(c).carrier, n.df(c).country]).sum(**agg_group_kwargs).T) # agg_group_kwargs not defined ? ##agatha

    p_nom_max_w = n.df(c).p_nom_max.div(n.df(c).weight).loc[ext_and_active.columns]
    p_nom_max_t = expand_series(p_nom_max_w, time_valid).T

    rhs = (p_nom_max_t.mul(ext_and_active)
           .groupby([n.df(c).carrier, n.df(c).country], axis=1)
           .max(**agg_group_kwargs))

    define_constraints(n, lhs, "<=", rhs, 'GlobalConstraint', 'res_limit')


# functions for extra functionalities -> added from pypsa-eur ##agatha
# add_BAU_constraints, add_SAFE_constraint, add_operational_reserve_margin_constraint
# line 473 - 530 -> otherwise functions not defined
def add_BAU_constraints(n, config):
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    lhs = (
        linexpr((1, get_var(n, "Generator", "p_nom")))
        .groupby(n.generators.carrier)
        .apply(join_exprs)
    )
    define_constraints(n, lhs, ">=", mincaps[lhs.index], "Carrier", "bau_mincaps")

def add_SAFE_constraints(n, config):
    peakdemand = (
        1.0 + config["electricity"]["SAFE_reservemargin"]
    ) * n.loads_t.p_set.sum(axis=1).max()
    conv_techs = config["plotting"]["conv_techs"]
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conv_techs"
    ).p_nom.sum()
    ext_gens_i = n.generators.query("carrier in @conv_techs & p_nom_extendable").index
    lhs = linexpr((1, get_var(n, "Generator", "p_nom")[ext_gens_i])).sum()
    rhs = peakdemand - exist_conv_caps
    define_constraints(n, lhs, ">=", rhs, "Safe", "mintotalcap")

def add_operational_reserve_margin_constraint(n, config):
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    reserve = get_var(n, "Generator", "r")
    lhs = linexpr((1, reserve)).sum(1)

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        renewable_capacity_variables = get_var(n, "Generator", "p_nom")[
            vres_i.intersection(ext_i)
        ]
        lhs += linexpr(
            (-EPSILON_VRES * capacity_factor, renewable_capacity_variables)
        ).sum(1)

    # Total demand at t
    demand = n.loads_t.p_set.sum(1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    define_constraints(n, lhs, ">=", rhs, "Reserve margin")



### all added AM #####################################################

def calculate_and_print_emissions_and_taxes(n):

    emission_prices = snakemake.config['costs']['emission_prices']

    ep = (pd.Series(emission_prices).rename(lambda x: x+'_emissions') * n.carriers).sum(axis=1)
    n.generators['marginal_cost'] += n.generators.carrier.map(ep)
    n.storage_units['marginal_cost'] += n.storage_units.carrier.map(ep)

    # Calculate total emissions
    total_emissions = (n.generators_t.p * n.generators.carrier.map(ep)).sum().sum()
    carbon_taxes = total_emissions/1000 * 539  # 539 ZAR per ton - 30 USD per ton
    #print(f"Total CO2 emissions: {total_emissions} tons")
    #print(f"Carbon taxes: {carbon_taxes} USD")

     # Convert total emissions to megatonnes (Mt) - for printing
    total_emissions_mt = total_emissions /1e6
    # Convert carbon taxes to million USD (M USD) - for printing
    carbon_taxes_musd = (total_emissions_mt * 30) # USD
    
    print(f"Total CO2 emissions: {total_emissions_mt} Mt")
    print(f"Carbon taxes: {carbon_taxes_musd} M USD")
    
    # Save the emissions and carbon taxes to a CSV file
    df = pd.DataFrame({"co2_emissions_mt": [total_emissions_mt], "carbon_taxes_musd": [carbon_taxes_musd]})
    csv_output = "results/networks/emissions_taxes.csv"
    df.to_csv(csv_output, index=False)
    print(f"Emissions and carbon taxes saved to {csv_output}")

    return total_emissions_mt, carbon_taxes_musd

def add_emission_prices(n, emission_prices=None, exclude_co2=False):
    if emission_prices is None:
        emission_prices = snakemake.config['costs']['emission_prices']
    if exclude_co2: 
        emission_prices.pop('co2')
    ep = (pd.Series(emission_prices).rename(lambda x: x+'_emissions') * n.carriers).sum(axis=1)
    n.generators['marginal_cost'] += n.generators.carrier.map(ep)
    n.storage_units['marginal_cost'] += n.storage_units.carrier.map(ep)


def add_carbontax_constraints(n, year=2030, additional_investment=0, base_investment=0):
    invest_dict = {'onwind': 12708, 'solar': 8619}  # ZAR/kWel
    renewable_carriers = ['onwind', 'solar']

    add_generators = n.generators[(n.generators['carrier'].isin(renewable_carriers))
                                  & (n.generators.build_year == year)]
    
    add_generators['investment_cost'] = add_generators['carrier'].map(invest_dict) * 1000  # ZAR/MW

    if add_generators.empty or ('Generator', 'p_nom') not in n.variables.index:
        return

    generators_p_nom = get_var(n, "Generator", "p_nom")

    lhs = linexpr((add_generators['investment_cost'], generators_p_nom[add_generators.index])).sum()

    total_investment = base_investment+ (additional_investment)  #* 1e6) # base investment + additional investment in ZAR
    define_constraints(n, lhs, ">=", total_investment, 'Generator-Storage', 'additional_carbontax_investment')


def reinvest_carbon_taxes(n, config, opts, base_investment):
    tolerance = 0.00001  # T0.01 olerance for convergence in M ?
    iteration = 0

    # Initial solve to get the initial carbon taxes
    n, initial_emissions, initial_carbon_taxes, base_investment = solve_network(n, config, opts, base_investment=base_investment)
    previous_carbon_taxes = initial_carbon_taxes

    while True:
        print(f"Iteration {iteration}: Reinvesting {previous_carbon_taxes} M USD in renewable energy.")
        additional_investment = previous_carbon_taxes  # Reinvest carbon taxes

        # Solve the network with the additional investment - fourth value not needed
        n, emissions, carbon_taxes, _ = solve_network(
            n, config=config, opts=opts, additional_investment=additional_investment, base_investment=base_investment)

        # Print the results of the current iteration
        print(f"Iteration {iteration}: Emissions = {emissions} tons, Carbon Taxes = {carbon_taxes} M USD")


        # Append the results of the current iteration to a DataFrame
        #df_iterations = df_iterations.append({"iteration": iteration, "emissions_tons": emissions, "carbon_taxes_zar": carbon_taxes}, ignore_index=True)

        # # Save the emissions and carbon taxes for each iteration to a CSV file
        # csv_output_iterations = "results/networks/emissions_taxes_iterations.csv"
        # df_iterations.to_csv(csv_output_iterations, index=False)
        # print(f"Emissions and carbon taxes for each iteration saved to {csv_output_iterations}")


        # Check for convergence
        #print(f"Emissions: {emissions} Mt, Carbon Taxes: {carbon_taxes} M USD")
        if abs(carbon_taxes - previous_carbon_taxes) < tolerance:
            print("Convergence reached.")
            break

        previous_carbon_taxes = carbon_taxes
        iteration += 1

    return n


########################


def extra_functionality(n, snapshots, additional_investment=0, base_investment=0):
    """
    Collects supplementary constraints which will be passed to ``pypsa.linopf.network_lopf``.
    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "BAU" in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, snapshots, config)
    if "SAFE" in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, snapshots, config)
    if "CCL" in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, snapshots,config)
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin_constraint(n, snapshots, config) #added _constraint
    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, snapshots, o)
    min_capacity_factor(n,snapshots)
    define_storage_global_constraints(n, snapshots)
    reserves(n,snapshots)
    # added AM constraints
    #add_carbontax_contraints1(n)
    #add_carbon_taxes(n)
    add_carbontax_constraints(n, year=2030, additional_investment=additional_investment,base_investment=base_investment)
    add_emission_prices(n)


def solve_network(n, config, opts="",additional_investment=0, base_investment=0, **kwargs):
    solver_options = config["solving"]["solver"].copy()
    solver_name = solver_options.pop("name")
    cf_solving = config["solving"]["options"]
    track_iterations = cf_solving.get("track_iterations", False)
    min_iterations = cf_solving.get("min_iterations", 4)
    max_iterations = cf_solving.get("max_iterations", 6)

    multi_investment_periods=isinstance(n.snapshots, pd.MultiIndex)
    multi_investment_periods=False

    #  only consider investments until 2030
    # wished_sn = n.snapshots[n.snapshots.get_level_values(0)<=2030]
    # n.set_snapshots(wished_sn)

    # Manage the GlobalConstraint - added AM - doubled GC through looping
    constraint_name = "CO2Limit2030"
    if constraint_name in n.global_constraints.index:
        n.global_constraints.drop(constraint_name, inplace=True)
    n.add("GlobalConstraint",
          constraint_name,
          carrier_attribute="co2_emissions",
          sense="<=",
          investment_period=2030,
          constant=275e6)  # max CO2 2030 96e6, IRP 2030 275e6

    # n.add("GlobalConstraint",
    #       "CO2Limit2030",
    #       carrier_attribute="co2_emissions",
    #       sense="<=",
    #       investment_period=2030,
    #       constant=275e6) #max CO2 2030 96e6, IRP 2030 275e6


    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if (snakemake.wildcards.regions=='RSA') | (cf_solving.get("skip_iterations", False)):
        network_lopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            multi_investment_periods=multi_investment_periods,
            extra_functionality=extra_functionality,
            **kwargs
        )
    else:
        ilopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            track_iterations=track_iterations,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            multi_investment_periods=multi_investment_periods,
            extra_functionality=extra_functionality,
            **kwargs
        )

    # Calculate and print emissions and carbon taxes after solving the network - Agatha
    #calculate_and_print_emissions_and_taxes(n)

    # Calculate and print emissions and carbon taxes after solving the network - added AM
    emissions, carbon_taxes = calculate_and_print_emissions_and_taxes(n)

    # Calculate the base investment - added AM
    if base_investment == 0:
        base_investment = (n.generators.p_nom_opt * n.generators.capital_cost).sum()

    return n, emissions, carbon_taxes, base_investment  # added AM

    #return n



#%%
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'solve_network',
            **{
                'model_file':'val-LC-UNC',
                'regions':'27-supply',
                'resarea':'redz',
                'll':'copt',
                'opts':'Co2L-3H',
                'attr':'p_nom'
            }
        )
    configure_logging(snakemake)

    tmpdir = snakemake.config["solving"].get("tmpdir")
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    opts = snakemake.wildcards.opts.split("-")
    solve_opts = snakemake.config["solving"]["options"]

    fn = getattr(snakemake.log, "memory", None)
    with memory_logger(filename=fn, interval=30.0) as mem:
        n = pypsa.Network(snakemake.input[0])
        n.set_snapshots(n.snapshots[n.snapshots.get_level_values(0)==2030])
        n.global_constraints = n.global_constraints[n.global_constraints.index.str.contains("2030")]
        if snakemake.config["augmented_line_connection"].get("add_to_snakefile"):
            n.lines.loc[
                n.lines.index.str.contains("new"), "s_nom_min"
            ] = snakemake.config["augmented_line_connection"].get("min_expansion")
        n = prepare_network(n, solve_opts)

        # n = solve_network(
        #     n,
        #     config=snakemake.config,
        #     opts=opts,
        #     solver_dir=tmpdir,
        #     solver_logfile=snakemake.log.solver,
        #     #keep_references=True, #only for debugging when needed
        #)


        # add base investment and pass to reinvest_carbon_taxes - added AM
        n, initial_emissions, initial_carbon_taxes, base_investment = solve_network(
            n,
            config=snakemake.config,
            opts=opts,
            solver_dir=tmpdir,
            solver_logfile=snakemake.log.solver,
        )


        # Run the reinvestment loop with the calculated base investment
        n = reinvest_carbon_taxes(n, snakemake.config, opts, base_investment)


        # # Perform the initial solve - added AM
        # n, initial_emissions, initial_carbon_taxes = solve_network(
        #     n,
        #     config=snakemake.config,
        #     opts=opts,
        #     solver_dir=tmpdir,
        #     solver_logfile=snakemake.log.solver,
        # )

        # # Run the reinvestment loop - added AM
        # n = reinvest_carbon_taxes(n, snakemake.config, opts)

        n.export_to_netcdf(snakemake.output[0])
    logger.info("Maximum memory usage: {}".format(mem.mem_usage))