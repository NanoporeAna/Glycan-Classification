import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as ds
import pandas as pd
import numpy as np
res = []
df = pd.read_json('C:\\Users\\Lucifer_Chen\\PycharmProjects\\HSQCNMR\\data\\HSQC_shuffle.json')
ac_wt = df['molecular_weight'][12123:12223].values
smiles_wt = df['smiles'][12123:12223]
rd_weight = []
for index, smile in enumerate(smiles_wt):
    wt = ds.ExactMolWt(Chem.MolFromSmiles(smile))
    rd_weight.append(wt)
    res.append(abs(wt-ac_wt[index]))

data = {'smiles': smiles_wt.values, 'molecular_weight': ac_wt, 'rdkit_weight': rd_weight}
t = pd.DataFrame(data)
t.to_excel('checkCoCoNut100.xlsx')
