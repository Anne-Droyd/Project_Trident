

import pandas as pd
from main_dir.Methods.Data_Options import data_options

data_ops = data_options(type="VR_DATA")
data = data_ops.get_data()


df1 = data[data.isna().any(axis=1)]
data = data.dropna()
# for index, row in df1.iterrows():
#     print(row)
# print(data.info())
# print(data.isnull().sum())

earth_mass = 5.97e24
phobos_mass = 1.06e16
for index, row in data.iterrows():
    mass_frac = row["m_core"] + row["matm"] + row["mdeep"]
    ice_mass = (row["zatm0"]*row["matm"] + row["zdeep0"]*row["mdeep"])
    rock_mass = (row["zatm1"]*row["matm"] + row["zdeep1"]*row["mdeep"])
    mass_residual_from_z = (1 - mass_frac) * row["mass"]
    mass_planet = row["mass"]
    mass_core = row["m_core"]*row["mass"]
    mass_planet_no_h_he = mass_core + ice_mass + rock_mass
    h_he_mass = mass_planet-mass_planet_no_h_he
    if mass_residual_from_z <= mass_planet_no_h_he:
        print(f"ice mass {ice_mass:.3}M☉")
        print(f"rock mass {rock_mass:.3}M☉")
        print(f"H/He mass {h_he_mass:.4}M☉")
        print(f"core mass {mass_core:.4}M☉")
        print(f"planet mass {mass_planet:.4}M☉")
        print(f"mass residual {mass_residual_from_z:.4}M☉")
    # if mass_frac <= 1:
    #     mass = (1-mass_frac)*row["mass"]*5.97e24
    #     phobos=mass/1.06e16
    #     if phobos <= 1.0e1:
    #         print(f"{mass:.2} kilos discrepancy, {phobos:.2} phoboseses")