

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

data["ice_mass"] = data["zatm0"] * data["matm"] + data["zdeep0"] * data["mdeep"]
data["rock_mass"] = data["zatm1"] * data["matm"] + data["zdeep1"] * data["mdeep"]
data["h_he_mass"] = data["mass"] - (data['m_core'] + data["rock_mass"] + data["ice_mass"])
columns = data.columns.tolist()

file_outname= "C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/Data/default.dat"
data.to_csv(file_outname,index=False,sep="\t",columns=columns,header=True)

for index, row in data.iterrows():
    mass_frac = row["m_core"] + row["matm"] + row["mdeep"]
    ice_mass = (row["zatm0"]*row["matm"] + row["zdeep0"]*row["mdeep"])
    rock_mass = (row["zatm1"]*row["matm"] + row["zdeep1"]*row["mdeep"])
    mass_residual = (1 - mass_frac) * row["mass"]
    mass_planet = row["mass"]
    mass_core = row["m_core"]*row["mass"]
    mass_planet_no_h_he = mass_core + ice_mass + rock_mass
    h_he_mass = mass_planet-mass_planet_no_h_he
    # if mass_planet_no_h_he:
    #     print(f"ice mass {ice_mass:.3}M☉")
    #     print(f"rock mass {rock_mass:.3}M☉")
    #     print(f"H/He mass {h_he_mass:.4}M☉")
    #     print(f"core mass {mass_core:.4}M☉")
    #     print(f"planet mass {mass_planet:.4}M☉")
    #     print(f"mass residual {mass_residual:.4}M☉")