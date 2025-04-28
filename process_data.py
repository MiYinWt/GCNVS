import pandas as pd


df = pd.read_csv('./data/BindingDB_All.tsv',sep='\t', low_memory=False)

smiles_col = 'Ligand SMILES'
TargetName_col = 'Target Name'
IC50_col = 'IC50 (nM)'

df = df.drop_duplicates(subset=['Ligand SMILES'], keep='first')

#Filter Targets
filtered_data = df[df[TargetName_col].str.contains('Fibroblast growth factor receptor 1', na=False)]

filtered_data = filtered_data.dropna(subset=['IC50 (nM)'])

filtered_data[IC50_col] = filtered_data[IC50_col].str.extract(r'(\d+\.?\d*)', expand=False)

filtered_data[IC50_col] = pd.to_numeric(filtered_data[IC50_col], errors='coerce')

filtered_data['Label'] = filtered_data[IC50_col].apply(lambda x: 1 if x <= 100 else 0)



result_data = filtered_data[['BindingDB Reactant_set_id','Ligand SMILES', 'Target Name', 'IC50 (nM)', 'Label']]

output_file = 'filtered_data.csv'  # 输出文件名
result_data.to_csv(output_file, index=False)  # 保存为 CSV 文件，不保存索引

print(f"处理完成，结果已保存到 {output_file}")
