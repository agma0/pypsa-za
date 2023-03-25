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

# types from config
#   extendable_carriers:
#     Renewables: ['onwind', solar]
#     Generator: [CCGT, OCGT, coal, nuclear, hydro-import]
#     StorageUnit: [battery, PHS]


# ZAR 1,4 billion in 2021 with ZAR 122/tCO2e ->1,4e9
# 8 billion USD 8e9 bis 2030 -> avergae exchange 2022 - 17.673 -> 141.384 billion ZAR
# 141.384 1e9 ZAR

add_carbon_investment = 141.384e9

def add_carbontax_contraints(n, year=2030):
    renewable_carriers = ['onwind', 'solar', 'CSP', 'biomass', 'hydro'] # in config.yaml deklariert #hydroimport???
    add_generators = n.generators[(n.generators['carrier'].isin(renewable_carriers))
                                  & (n.generators.build_year==year)]
    add_storage_units = n.storage_units[(n.storage_units['carrier'] == 'PHS')
                                        & (n.storage_units.build_year==year)]

    if add_generators.empty and add_storage_units.empty or ('Generator', 'p_nom') not in n.variables.index:
        return

    generators_p_nom = get_var(n, "Generator", "p_nom")

    stores_p_nom = get_var(n, "StorageUnit", "p_nom")
    lhs = linexpr((add_generators['capital_cost'], generators_p_nom[add_generators.index])).sum()
    lhs += linexpr((add_storage_units['capital_cost'],stores_p_nom[add_storage_units.index])).sum()

    define_constraints(n, lhs, ">=", add_carbon_investment, 'Generator-Storage', 'additional_carbontax_investment')

### ideas
# define 25% , 50%, 75%, 100% of revenues - show scenarios
# climate goals 2030 - 400 MtCO2e ? -> change CO2 limit in config.yaml line 69
# end load shedding? changes in loadshedding (2030) if revenues are invested in RE?

# add new data SACAD/SAPAD? 2022?

# where to provide free solar geysers? -> from household data!
