import pandas as pd

file_path = "/home/matt/project/project_files/Exo_1040.dat"
df = pd.read_csv(file_path,delim_whitespace=True)

df["test"]=0.5*df["mass"]
df.to_csv("/home/matt/project/project_files/Exo_1040.dat",sep="\t",index=False)

