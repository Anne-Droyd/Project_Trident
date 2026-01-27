import pandas as pd
from main_dir.Methods.Data_Options import data_options

data_ops=data_options(type="VR_DATA",default_data=False)
original_data = data_ops.get_data()
updated_data = data_ops.get_data()

original_data = original_data.drop(columns='test')
# updated_data = updated_data.drop(columns=['matm','mdeep'])

frames = [original_data,updated_data]
merged = pd.concat(frames)

merged = merged.rename(columns={'req':'radius','Teq':'temp'})

print(merged.head)


path = 'C:/Users/Matth/Documents/Leiden University/Project/Masters Project Main/Data/'
merged_file = path + 'merged.dat'
merged.to_csv(merged_file,sep='\t',index=False,header=True)
