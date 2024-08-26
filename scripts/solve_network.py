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
            marginal_cost=1e5,#load_shedding,
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
           .groupby([n.df(c).carrier, n.df(c).country]).sum(**agg_group_kwargs).T)

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


####

def add_emission_prices(n, emission_prices=None):
    if emission_prices is None:
        emission_prices = snakemake.config['costs']['emission_prices']

    # Set the marginal costs of solar and wind
    #n.generators.loc[n.generators.carrier == 'solar', 'marginal_cost'] = 0.01
    #n.generators.loc[n.generators.carrier == 'onwind', 'marginal_cost'] = 0.015

    ep = (
        pd.Series(emission_prices).rename(lambda x: x + "_emissions")
        * n.carriers.filter(like="_emissions")
    ).sum(axis=1)

    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators["marginal_cost"] += gen_ep.fillna(0)
    
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units["marginal_cost"] += su_ep.fillna(0)

    # Print marginal costs immediately after application
    print("\033[91mMarginal costs after applying emission prices:\033[0m")
    print("\033[91m", n.generators[["carrier", "marginal_cost"]].groupby("carrier").mean(), "\033[0m")


    return ep  # Ensure that `ep` is returned and is not None

#####




# Define max additional investment in ZAR - CO2 tax revenue 2030 with 30US$

# ZAR 1,4 billion in 2021 with ZAR 122/tCO2e ->1,4e9
# 8 billion USD 8e9 bis 2030 -> 2030: 3.19 billion US$ - average exchange 2022 - 17.673
# 56.377 1e9 ZAR

carbon_investment = 556.377e9 #+base capcost*lifetime = invest
#onwind= 900.61e6 * 20 , solar=651.90e6 * 25  - only solar and wind!!!   CSP=356.46e6, biomass=109.11e6, hydro=0, PHS=230.58e6 -> 
rebase_investment = 606.355e9  # from base model capital costs of renewables 34309.7e6 USD * 17.673ZAR/USD = 606355.33e6
total_investment = carbon_investment + rebase_investment

# ### new constraint - total investment of carbon tax revenues in 2030 into renewables
# def add_carbontax_constraints(n, year=2030):
#     invest_dict = {'onwind': 12708, 'solar': 8619} #, 'CSP': 0, 'biomass': 20232.73107, 'hydro': 2000, 'PHS': 2000} # ZAR/kWel
#     renewable_carriers = ['onwind', 'solar'] #, 'CSP', 'biomass', 'hydro']

#     add_generators = n.generators[(n.generators['carrier'].isin(renewable_carriers))
#                                   & (n.generators.build_year==year)]
#     #add_storage_units = n.storage_units[(n.storage_units['carrier'] == 'PHS')
#     #                                    & (n.storage_units.build_year==year)]
    
#     # Safely assign investment_cost using .loc
#     add_generators.loc[:, 'investment_cost'] = add_generators['carrier'].map(invest_dict) * 1000  # ZAR/MW

#     #add_generators['investment_cost'] = add_generators['carrier'].map(invest_dict) *1000 #ZAR/MW
#     #add_storage_units['investment_cost'] = add_storage_units['carrier'].map(invest_dict) *1000

#     #and add_storage_units.empty 
#     if add_generators.empty or ('Generator', 'p_nom') not in n.variables.index:
#         return

#     generators_p_nom = get_var(n, "Generator", "p_nom")
#     #stores_p_nom = get_var(n, "StorageUnit", "p_nom")

#     lhs = linexpr((add_generators['investment_cost'], generators_p_nom[add_generators.index])).sum()
#     #lhs += linexpr((add_storage_units['investment_cost'],stores_p_nom[add_storage_units.index])).sum()

#     define_constraints(n, lhs, ">=", total_investment, 'Generator-Storage', 'additional_carbontax_investment')

#     print("\033[91mReinvestment in Renewable Energies!\033[0m")

def add_carbontax_constraints(n, year=2030, total_investment=1e6):  # Example total investment
    invest_dict = {'onwind': 12708, 'solar': 8619}  # ZAR/kWel for onshore wind and solar
    renewable_carriers = ['onwind', 'solar']

    # Retrieve the renewable generators for the specific year
    add_generators = n.generators[(n.generators['carrier'].isin(renewable_carriers)) &
                                  (n.generators['build_year'] == year)]

    # Skip if there are no generators or investment data
    if add_generators.empty or ('Generator', 'p_nom') not in n.variables.index:
        print("No generators or capacity variables found for the year specified.")
        return

    # Add the investment cost to the relevant generators
    # This is where we make sure that the change affects the original DataFrame
    n.generators.loc[add_generators.index, 'investment_cost'] = add_generators['carrier'].map(invest_dict) * 1000  # ZAR/MW

    # Get the generator nominal power variables
    generators_p_nom = get_var(n, "Generator", "p_nom")

    # Define the linear expression for the constraint (investment_cost * nominal capacity)
    lhs = linexpr((n.generators.loc[add_generators.index, 'investment_cost'], 
                   generators_p_nom[add_generators.index])).sum()

    # Apply the constraint ensuring total investment is met
    define_constraints(n, lhs, ">=", total_investment, 'Generator', 'additional_carbontax_investment')

    print("\033[91mReinvestment in Renewable Energies successfully applied!\033[0m")


######### Everything for reinvestment loop
    
def calculate_and_print_emissions_and_taxes(n, iteration, ep=None):
    
    # Only apply emission prices during the first iteration
    if iteration == 0:
        ep = add_emission_prices(n)
    elif ep is None:
        raise ValueError("Emission prices (ep) must be provided for iterations after the first one.")


    # Calculate total emissions by summing the products of generation and emission intensity
    total_emissions = (n.generators_t.p * n.generators.carrier.map(ep)).sum().sum()
    total_emissions_mt = total_emissions / 1e6  # Convert emissions to Megatonnes

    # Apply carbon taxes (assuming $30/ton CO2 and ZAR equivalent for the taxes)
    carbon_taxes_mzar = (total_emissions_mt * 560)  # ZAR tax on emissions
    carbon_taxes_musd = (total_emissions_mt * 30)   # USD tax on emissions

    # Print the results for monitoring purposes
    print(f"Total CO2 emissions: {total_emissions_mt} Mt")
    print(f"Carbon taxes: {carbon_taxes_mzar} M ZAR")
    print(f"Carbon taxes: {carbon_taxes_musd} M USD")

    return total_emissions_mt, carbon_taxes_mzar, ep


# def apply_reinvestment_to_renewables(n, reinvestment_amount):
#     invest_dict = {'onwind': 12708, 'solar': 8619}  # ZAR/kWel for onshore wind and solar
#     renewable_carriers = ['onwind', 'solar']
    
#     # Retrieve the renewable generators that are extendable
#     add_generators = n.generators[(n.generators['carrier'].isin(renewable_carriers)) & 
#                                   (n.generators['p_nom_extendable'])]
    
#     # Skip if there are no extendable renewable generators
#     if add_generators.empty:
#         print("No extendable renewable generators found.")
#         return
    
#     # Safely map the investment costs based on carrier type directly in the original DataFrame
#     n.generators.loc[add_generators.index, 'investment_cost'] = add_generators['carrier'].map(invest_dict) * 1000  # ZAR/MW
    
#     # Distribute total investment proportionally to the generators
#     total_generators = len(add_generators)
    
#     if total_generators == 0:
#         print("No valid renewable generators found.")
#         return
    
#     # Loop through the renewable generators and apply the reinvestment
#     for idx, gen in add_generators.iterrows():
#         # Calculate new capacity proportionally based on investment cost
#         additional_capacity = reinvestment_amount / total_generators / gen['investment_cost']
        
#         # Apply the additional capacity directly to the original DataFrame
#         n.generators.at[idx, 'p_nom'] += additional_capacity
    
#     print(f"Reinvestment of {reinvestment_amount} ZAR applied to renewable generators.")




def apply_reinvestment_to_renewables(n, reinvestment_amount):

    invest_dict = {'onwind': 12708, 'solar': 8619}  # ZAR/kWel for onshore wind and solar
    renewable_carriers = ['onwind', 'solar']
    
    # Retrieve generators to which the reinvestment will be applied
    add_generators = n.generators[(n.generators['carrier'].isin(renewable_carriers)) & (n.generators.p_nom_extendable)]
    
    # Skip if there are no extendable renewable generators
    if add_generators.empty:
        print("No extendable renewable generators found.")
        return
    
    # Map the investment costs based on carrier type
    add_generators['investment_cost'] = add_generators['carrier'].map(invest_dict) * 1000  # ZAR/MW
    
    # Distribute total investment proportionally to the generators
    for idx, gen in add_generators.iterrows():
        # Calculate new capacity proportionally based on investment cost
        additional_capacity = total_investment / len(add_generators) / gen['investment_cost']
        n.generators.at[idx, 'p_nom'] += additional_capacity
    
    print(f"Reinvestment of {total_investment} ZAR applied to renewable generators.")

#####

def one_time_investment(n, base_investment):
    # Calculate emissions and carbon taxes after the first solve
    total_emissions, carbon_taxes_mzar, _ = calculate_and_print_emissions_and_taxes(n, iteration=0)

    # Calculate total investment
    additional_investment = carbon_taxes_mzar
    total_investment = base_investment + additional_investment

    # Apply the one-time investment
    apply_reinvestment_to_renewables(n, total_investment)

    print(f"One-time investment applied: Base Investment = {base_investment} ZAR, "
          f"Additional Investment = {additional_investment} ZAR, "
          f"Total Investment = {total_investment} ZAR")

    return n

####


def reinvestment_loop(n, base_investment, max_iterations=15, convergence_threshold=1e-3):

    previous_investment = base_investment
    iteration = 0
    ep = None  # Initialize ep as None

    while iteration < max_iterations:
        # Calculate emissions and carbon taxes for this iteration
        total_emissions, carbon_taxes_mzar, ep = calculate_and_print_emissions_and_taxes(n, iteration,ep)
        
        # Calculate the total investment for this iteration
        total_investment = base_investment + carbon_taxes_mzar
        
        # Apply the investment to renewables
        apply_reinvestment_to_renewables(n, total_investment)

        # Print iteration details for debugging
        print(f"Iteration {iteration}: Total Investment = {total_investment} ZAR, Total Emissions = {total_emissions} Mt")
        
        # Check if the investment has converged
        if abs(total_investment - previous_investment) < convergence_threshold:
            print(f"Convergence reached at iteration {iteration}.")
            break
        
        # Update previous investment and increment iteration
        previous_investment = total_investment
        iteration += 1
    
    if iteration == max_iterations:
        print(f"Max iterations reached. Reinvestment may not have fully converged.")
    
    return n


######

def extra_functionality(n, snapshots):
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
    #add_emission_prices(n) - not here but in the solve_network to apply it before optimazation line 718
    #add_carbontax_constraints(n)



def solve_network(n, config, opts="", **kwargs):
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
          constant=275e6) #max CO2 2030 100e6, IRP 2030 275e6
    
    ## Apply emission prices BEFORE the optimization starts - added AM - UNCOMMENT IN LOOP!!!!
    #print("\033[91mEmission prices added!\033[0m")
    #add_emission_prices(n)


    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    # First solve the network
    print("\033[91mStarting first network solve...\033[0m")

    # Initialize iteration variable - for loop
    iteration = 0  


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
    

    return n


#%%
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'solve_network',
            **{
                'model_file':'val-2Gt-IRP',
                'regions':'27-supply',
                'resarea':'redz',
                'll':'copt',
                'opts':'Co2-2190H',
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

    # NORMAL RUN
    print("\033[91mNormal run!\033[0m")
    n = solve_network(
        n,
        config=snakemake.config,
        opts=opts,
        solver_dir=tmpdir,
        solver_logfile=snakemake.log.solver,
        #keep_references=True, #only for debugging when needed
    )

    # Print the total CO2 emissions
    total_emissions = (n.generators_t.p * n.generators.carrier.map(n.carriers['co2_emissions'])).sum().sum()
    total_emissions_mt = total_emissions / 1e6  # Convert emissions to Megatonnes
    print(f"Total CO2 Emissions: {total_emissions_mt} Mt")

    # REINVESTMENT RUN -> UNCOMMENT WHEN NOT NEEDED
    # Extract the base investment and emissions after the first solve
    base_investment = (n.generators.p_nom_opt * n.generators.capital_cost).sum()

    #ONE TIME INVESTMENT - UNCOMMENT IF NOT NEEDED
    # Now run the one time reinvestment after the first normal run
    print("\033[91mStarting reinvestment loop...\033[0m")
    n = one_time_investment(n, base_investment=base_investment)

    # #INVESTMENT LOOP - UNCOMMENT IF NOT NEEDED
    # # Now run the reinvestment loop after the first normal run
    # print("\033[91mStarting reinvestment loop...\033[0m")
    # n = reinvestment_loop(n, base_investment=base_investment)

    # Solve the network again after reinvestment
    print("\033[91mSolving network after reinvestment...\033[0m")
    n = solve_network(
        n, 
        config=snakemake.config,
        opts=opts,
        solver_dir=tmpdir, 
        solver_logfile=snakemake.log.solver
    )

    # Print the total CO2 emissions
    total_emissions = (n.generators_t.p * n.generators.carrier.map(n.carriers['co2_emissions'])).sum().sum()
    total_emissions_mt = total_emissions / 1e6  # Convert emissions to Megatonnes
    print(f"Total CO2 Emissions: {total_emissions_mt} Mt")
    #


    n.export_to_netcdf(snakemake.output[0])
    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
