import pandas as pd

df = pd.read_csv('./data/BindingDB_All.tsv',sep='\t', low_memory=False)

smiles_col = 'Ligand SMILES'
TargetName_col = 'Target Name'
IC50_col = 'IC50 (nM)'


#Filter Targets
filtered_data = df[df[TargetName_col].str.contains('Fibroblast growth factor receptor 1', na=False)]

filtered_data = filtered_data.dropna(subset=['IC50 (nM)'])

filtered_data[IC50_col] = filtered_data[IC50_col].str.extract(r'(\d+\.?\d*)', expand=False)

filtered_data[IC50_col] = pd.to_numeric(filtered_data[IC50_col], errors='coerce')

filtered_data['Label'] = filtered_data[IC50_col].apply(lambda x: 1 if x <= 100 else 0)

filtered_data = filtered_data.drop_duplicates(subset=['Ligand InChI Key'], keep='first')

result_data = filtered_data[['BindingDB Reactant_set_id','Ligand SMILES', 'Ligand InChI Key','Target Name', 'IC50 (nM)', 'Label']]

output_file = './data/filtered_data.csv'  
result_data.to_csv(output_file, index=False)  

print(f"The processing is completed, the filtered data is saved to {output_file}")


####Divide the dataset

result_data = pd.read_csv('./data/filtered_data.csv')

active_data = result_data[result_data['Label'] == 1]

inactive_data = result_data[result_data['Label'] == 0]

print(f"Active data count: {len(active_data)}")
print(f"Inactive data count: {len(inactive_data)}")


# train_dataset
train_active = active_data.sample(frac=0.6)
train_inactive = inactive_data.sample(frac=0.6)

active_data.drop(train_active.index, inplace=True)
inactive_data.drop(train_inactive.index, inplace=True)

train_dataset = pd.concat([train_active, train_inactive], ignore_index=True)
train_dataset.to_csv('./data/train.csv', index=False) 

#val_dataset
val_active = active_data.sample(frac=0.5)
val_inactive = inactive_data.sample(frac=0.5)

active_data.drop(val_active.index, inplace=True)
inactive_data.drop(val_inactive.index, inplace=True)

val_dataset = pd.concat([val_active, val_inactive], ignore_index=True)
val_dataset.to_csv('./data/val.csv', index=False) 

#test_dataset
test_dataset = pd.concat([active_data, inactive_data], ignore_index=True)
test_dataset.to_csv('./data/test.csv', index=False) 
