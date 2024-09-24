# %%
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Descriptors import ExactMolWt, NumHAcceptors, NumHDonors, NumRotatableBonds
import sys
from model_cvae import *
from utils import *

# %%
# Setting
work_dir = Path(__file__)
root_dir = work_dir.parents[0]

sys.path.append(str(root_dir))



parser = argparse.ArgumentParser(description='Model')
parser.add_argument('--dataset_dir', type=str, default=f'{root_dir}/data', help='Dataset directory')
parser.add_argument('--images_dir', type=str, default=f'{root_dir}/images', help='Images directory')
parser.add_argument('--save_dir', type=str, default=f'{root_dir}/save', help='Save directory')
parser.add_argument('--gen_dir', type=str, default=f'{root_dir}/generated', help='Generated directory')
parser.add_argument('--grammar', type=str, default='smiles', help='smiles or selfies')
parser.add_argument('--target', type=str, default='cdk2', help='cdk2, gamma, dpp4, initial')
args, _ = parser.parse_known_args()


# Load data
prop_file = f"{args.gen_dir}/generated_molecules_{args.grammar}_{args.target}.txt"
df_1 = pd.read_csv(prop_file, sep='\t').rename(columns={'smiles': 'SMILES'}).drop_duplicates()
print(f"The length of the generated dataset is: {len(df_1)}")


prop_file_2 = f"{args.dataset_dir}/smiles_{args.target}.txt" # Only datasets in the SMILES format
df_2 = pd.read_csv(prop_file_2, sep='\t').rename(columns={'CanonicalSMILES': 'SMILES'}).drop_duplicates()
print(f"The length of the original dataset is: {len(df_2)}")

df_2.rename(columns={'CanonicalSMILES': 'SMILES'}, inplace=True)
smiles_unique_all, perc_unique_all = fraction_unique(df_1['SMILES'], check_validity=True)
smiles_novel_all, perc_novel_all = novelty(smiles_unique_all, df_2['SMILES'])


mols = [
    Chem.MolFromSmiles(smiles) 
    for smiles in df_1['SMILES'] 
    if Chem.MolFromSmiles(smiles) is not None
]

df_1 = pd.DataFrame({'mols': mols})
df_1['SMILES'] = df_1['mols'].apply(Chem.MolToSmiles)
mss = [Chem.MolFromSmiles(smi) for smi in df_2['SMILES'] if Chem.MolFromSmiles(smi) is not None]


# %%
# Chemical properties comparison

# HBA
df_1['HBA'] = [NumHAcceptors(mol) for mol in tqdm(mols, desc='Computing HBA generated molecules')]
df_2['HBA'] = [NumHAcceptors(mol) for mol in tqdm(mss, desc='Computing HBA original molecules')]
data = count_and_plot(df_1, df_2, 'HBA', args)
compare_statistics(df_1, df_2, 'HBA')

# HBD
df_1['HBD'] = [NumHDonors(mol) for mol in tqdm(mols, desc='Computing HBD generated molecules')]
df_2['HBD'] = [NumHDonors(mol) for mol in tqdm(mss, desc='Computing HBD original molecules')]
data = count_and_plot(df_1, df_2, 'HBD', args)
compare_statistics(df_1, df_2, 'HBD')

# Rotatable bonds
df_1['RotB'] = [NumRotatableBonds(mol) for mol in tqdm(mols, desc='Computing RotB generated molecules')]
df_2['RotB'] = [NumRotatableBonds(mol) for mol in tqdm(mss, desc='Computing RotB original molecules')]
data = count_and_plot(df_1, df_2, 'RotB', args)
compare_statistics(df_1, df_2, 'RotB')

# Rings number
df_1['n_rings'] = [get_n_rings(mol) for mol in tqdm(mols, desc='Computing n_rings generated molecules')]
df_2['n_rings'] = [get_n_rings(mol) for mol in tqdm(mss, desc='Computing n_rings original molecules')]
data = count_and_plot(df_1, df_2, 'n_rings', args)
compare_statistics(df_1, df_2, 'n_rings')

# Molecules length
df_1['length'] = [len(i) for i in tqdm(df_1['SMILES'], desc='Computing length generated molecules')]
df_2['length'] = [len(i) for i in tqdm(df_2['SMILES'], desc='Computing length original molecules')]
compare_statistics(df_1, df_2, 'length')

# Rings length
df_1['len_rings'] = [ring_size(mol) for mol in tqdm(mols, desc='Computing len_rings generated molecules')]
df_2['len_rings'] = [ring_size(mol) for mol in tqdm(mss, desc='Computing len_rings original molecules')]
data = count_and_plot(df_1, df_2, 'len_rings', args)
compare_statistics(df_1, df_2, 'len_rings')

# MW 
df_1['MW'] = [ExactMolWt(mol) for mol in tqdm(mols, desc='Computing MW generated molecules')]
df_2['MW'] = [ExactMolWt(mol) for mol in tqdm(mss, desc='Computing MW original molecules')]
plot_properties(df_1, df_2, 'MW', args)
compare_statistics(df_1, df_2, 'MW')

# TPSA
df_1['TPSA'] = [CalcTPSA(mol) for mol in tqdm(mols, desc='Computing TPSA generated molecules')]
df_2['TPSA'] = [CalcTPSA(mol) for mol in tqdm(mss, desc='Computing TPSA original molecules')]
plot_properties(df_1, df_2, 'TPSA', args)
compare_statistics(df_1, df_2, 'TPSA')

# LogP
df_1['LogP'] = [MolLogP(mol) for mol in tqdm(mols, desc='Computing LogP generated molecules')]
df_2['LogP'] = [MolLogP(mol) for mol in tqdm(mss, desc='Computing LogP original molecules')]
plot_properties(df_1, df_2, 'LogP', args)
compare_statistics(df_1, df_2, 'LogP')

# Internal diversity
int_div = internal_diversity(df_1['SMILES'].unique())
print('The percentage of internal diversity is:', int_div*100)

# Fragments diversity
fragments = [fragmenter(mol) for mol in tqdm(mols, desc='Computing fragments')]
fragments = list(set([item for sublist in fragments for item in sublist]))
print('Number of fragments unique: ', len(fragments))

# Scaffold
scaffolds = [compute_scaffold(mol) for mol in tqdm(mols, desc='Computing scaffold')]
scaffoldss = pd.DataFrame(scaffolds)
scaffoldss.value_counts().sort_values(ascending=False)
print ('Number of unique scaffolds: ', scaffoldss.nunique())

# QED vs SA
df_1['QED'] = [QED(m) for m in tqdm(mols, desc= 'Computing QED generated molecules')]
df_1['SA'] = [SA(m) for m in tqdm(mols, desc= 'Computing SA generated molecules')]
df_2['QED'] = [QED(m) for m in tqdm(mss, desc= 'Computing QED original molecules')]
df_2['SA'] = [SA(m) for m in tqdm(mss, desc= 'Computing SA original molecules')]

upper = []
for i in range(len(df_1)):
    if df_1['QED'][i]>=0.5 and df_1['SA'][i]<=5:
        upper.append(1)
    else:
        upper.append(0)

print('The number of molecules in the upper left corner is:', sum(upper))
print('The percentage of molecules in the upper left corner is:', sum(upper)/len(df_1)*100)

plt.rc('font', size=35)
plt.rc('axes', titlesize=35)     
plt.rc('axes', labelsize=35)     
plt.rc('xtick', labelsize=35)    
plt.rc('ytick', labelsize=35)    
plt.rc('legend', fontsize=35)    

plt.figure(figsize=(20, 10), dpi=300)
plt.scatter(df_1['SA'], df_1['QED'], label='novel', color='#FF8C00', alpha=0.5)
plt.scatter(df_2['SA'], df_2['QED'], label='original')
plt.xlabel('SA')
plt.ylabel('QED')
plt.title('SA vs QED')
plt.legend()
plt.savefig(f'{args.images_dir}/QED_SA_{args.grammar}_{args.target}.png')
# %%
# Final filtering
df_1=df_1[df_1['HBD'] < 10]
df_1=df_1[df_1['HBA'] < 10]
df_1=df_1[df_1['n_rings'] < 8]
df_1=df_1[df_1['RotB'] < 15]
df_1=df_1[df_1['len_rings'] < 9]

print('The number of final molecules is:', len(df_1))

df_1[['SMILES', 'MW', 'LogP', 'TPSA', 'HBA', 'HBD', 'RotB', 'len_rings', 'QED', 'SA']].to_csv('final_molecules.csv', index=False)

