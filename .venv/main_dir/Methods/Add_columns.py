import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path = "C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/Data/Exo_1040_copy.dat"

data=pd.read_csv(path,delimiter="\t")
print(data)
data["density"]=data["mass"]*(5.97219e27)/(4*np.pi*(data["req"]*6.378e8)**3)
print(data)
plt.hist(data["density"],bins=100)
plt.show()