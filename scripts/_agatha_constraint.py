# add to solve_network extra_functionalitiy line 475 when finished

from pypsa.linopf import (
    define_constraints,
    define_variables,
    get_var,
    ilopf,
    join_exprs,
    linexpr,
    network_lopf,
)

# Define max additional investment in ZAR - CO2 tax income 2030 with 30US$

# carrier types from model_file
# generator: coal, gas, nuclear, onwind, hydro, hydro-import, solar, CSP, biomass
# storage unit: PHS, battery
# ccgt, ocgt

# ZAR 1,4 billion in 2021 with ZAR 122/tCO2e -> hochskalieren! Prognose 2030?
max_add_carbon_investment = 1400000000

def add_carbontax_contraints(n):
    renewable_carriers = ['solar', 'onwind', 'CSP', 'biomass', 'hydro'] # in config.yaml deklariert #hydroimport???
    add_generators = n.generators[n.generators['carrier'].isin(renewable_carriers)]
    add_storage_units = n.storage_units[n.storage_units['carrier'] == 'PHS']

    if add_generators.empty and add_storage_units.empty or ('Generator', 'p_nom') not in n.variables.index:
        return

    generators_p_nom = get_var(n, "Generator", "p_nom")
    lhs = linexpr((add_generators['capital_cost'], generators_p_nom[add_generators.index])).sum()
    lhs += linexpr((add_storage_units['capital_cost'],)).sum()

    define_constraints(n, lhs, "<=", max_add_carbon_investment, 'Generator&Storage', 'additional_carbontax_investment')

### ideas
# define 25% , 50%, 75%, 100% of revenues - show scenarios
# climate goals 2030 - 400 MtCO2e ? -> change CO2 limit in config.yaml line 69
# end load shedding? changes in loadshedding (2030) if revenues are invested in RE?

# add new data SACAD/SAPAD? 2022?

# where to provide free solar geysers? -> from household data!
