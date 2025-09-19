

import pandas as pd
from main_dir.Methods.Data_Options import data_options

data_ops = data_options(type="VR_DATA")
data = data_ops.get_data()

print(data.info())
print(data.isnull().sum())