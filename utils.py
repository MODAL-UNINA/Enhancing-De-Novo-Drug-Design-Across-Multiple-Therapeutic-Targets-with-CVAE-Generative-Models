# %%

# Imports
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import matplotlib.ticker as ticker
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Descriptors, AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer

def convert_to_smiles(vector, char):
    """
    Convert a vector to a SMILES string
    """
    list_char = list(char)
    vector = vector.astype(int)
    return "".join(map(lambda x: list_char[x], vector)).strip()


def load_vocab(file, args):
    """
    Load vocabulary and labels
    """

    f = open(file)
    lines = f.read().split('\n')[:-1]
    lines = [l.split() for l in lines]
    lines = [l for l in lines]
    labels = [l[1:] for l in lines][1:]
    if args.grammar == 'smiles':
        vocab = {'C': 0, '=': 1, '(': 2, ')': 3, 'N': 4, 'O': 5, '1': 6, '2': 7, 
                '3': 8, '4': 9, 'F': 10, 'S': 11, '5': 12, 'Cl': 13, '[O-1]': 14, 
                '[NH1]': 15, 'Br': 16, '#': 17, '[N-1]': 18, '[N+1]': 19, '[NH1+1]': 20, 
                'I': 21, 'P': 22, '[S-1]': 23, '[NH2+1]': 24, '[S+1]': 25, 'B': 26, 
                '[NH1-1]': 27, '[Si]': 28, '[C-1]': 29, '[NH3+1]': 30, '[Se]': 31, 
                '[B-1]': 32, '[O+1]': 33, '[PH1]': 34, '[P+1]': 35, '[2H]': 36, 
                '[SH1+1]': 37, '[CH1-1]': 38, '[Se+1]': 39, '[OH1+1]': 40, '[S+2]': 41, 
                '[Te+1]': 42, '[Te]': 43, '[SH1]':44, '6':45}

    elif args.grammar == 'selfies':
        vocab = {"[C]": 0, "[Ring1]": 1, "[Ring2]": 2, "[Branch1]": 3, "[=Branch1]": 4, 
        "[#Branch1]": 5, "[Branch2]": 6, "[=Branch2]": 7, "[#Branch2]": 8, 
        "[O]": 9, "[N]": 10, "[=N]": 11, "[=C]": 12, "[#C]": 13, "[S]": 14, 
        "[P]": 15, "[O-1]": 16, "[2H]": 17, "[=N-1]": 18, "[S+2]": 19, 
        "[S-1]": 20, "[=S+1]": 21, "[#N]": 22, "[N-1]": 23, "[=SH1+1]": 24, 
        "[#N+1]": 25, "[=Ring1]": 26, "[N+1]": 27, "[#C-1]": 28, "[B-1]": 29, 
        "[Te]": 30, "[=Se+1]": 31, "[NH1]": 32, "[Se+1]": 33, "[P+1]": 34, 
        "[=Te+1]": 35, "[NH1-1]": 36, "[NH1+1]": 37, "[=NH2+1]": 38, "[OH1+1]": 39, 
        "[C-1]": 40, "[NH3+1]": 41, "[=O]": 42, "[=N+1]": 43, "[I]": 44, "[=NH1+1]": 45, 
        "[Se]": 46, "[B]": 47, "[Si]": 48, "[Br]": 49, "[Cl]": 50, "[=Ring2]": 51, 
        "[=I]": 52, "[PH1]": 53, "[S+1]": 54, "[F]": 55, "[=Se]": 56, "[NH2+1]": 57,
        "[=S]": 58, "[=P]": 59, "[=O+1]": 60, "[CH1-1]": 61, "[=OH1+1]": 62, "[O+1]": 63, "[SH1]":64}
    
    chars = list(vocab)
    chars += ('E',)  #End of smiles
    chars += ('X',)  #Start of smiles
    vocab['E'] = len(chars)-2
    vocab['X'] = len(chars)-1

    return vocab, chars, labels


# %%
def count_and_plot(df_1, df_2, col, args):
    """
    Count and plot the number of molecules for a given property
    """

    data_1 = df_1[col].value_counts().reset_index()
    data_1.columns = [col, 'Count']
    data_1['Dataset'] = 'Generated'

    data_2 = df_2[col].value_counts().reset_index()
    data_2.columns = [col, 'Count']
    data_2['Dataset'] = 'Original'

    data = pd.concat([data_1, data_2])

    plt.figure(figsize=(20, 10), dpi=300)  # Imposta il DPI a 300
    ax = sns.barplot(x=col, y='Count', hue='Dataset', data=data, log=True, palette={'Generated': 'deepskyblue', 'Original': 'salmon'})
    ax.set_xlabel(f'Number of {col}', fontsize=36)
    ax.set_ylabel('Count', fontsize=36)
    if col == 'HBA':
        plt.title(f'Number of Hydrogen Bond Acceptors ({[col]})', fontsize=36)
    if col == 'HBD':
        plt.title(f'Number of Hydrogen Bond Donors ({[col]})', fontsize=36)
    if col == 'n_rings':
        plt.title(f'Number of Rings ({[col]})', fontsize=36)
    if col == 'RotB':
        plt.title(f'Number of Rotatable Bonds ({[col]})', fontsize=36)
    if col == 'len_rings':
        plt.title(f'Length of Rings ({[col]})', fontsize=36)
    plt.legend(fontsize=36, loc='upper right')
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.savefig(f'{args.images_dir}/{col}_{args.grammar}_{args.target}.png')
    return data

def plot_properties(df_1, df_2, property, args):
    """
    Plot the distribution of a given property
    """

    plt.figure(figsize=(20, 10), dpi=300)
    ax = sns.kdeplot(df_1[property], fill=True, color='deepskyblue', label='Generated')
    sns.kdeplot(df_2[property], fill=True, color='salmon', label='Original')
    ax.set_xlabel(f'{property}', fontsize=36)
    ax.set_ylabel('Density', fontsize=36)
    if property == 'TPSA':
        plt.title(f'Topological Polar Surface Area ({[property]})', fontsize=36)
    if property == 'LogP':
        plt.title(f'Octanol-water partition coefficient ({[property]})', fontsize=36)
    if property == 'MW':
        plt.title(f'Molecular Weight ({[property]})', fontsize=36)
    plt.legend(fontsize=36)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=36)
    ax.tick_params(axis='y', labelsize=36)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x * 1e3)))
    ax.set_ylabel('Density', fontsize=36)
    ax.annotate(r'$10^{-3}$', xy=(0, 1), xytext=(-40, 20), textcoords='offset points', 
            ha='center', va='bottom', fontsize=36, annotation_clip=False, xycoords='axes fraction')
    plt.savefig(f'{args.images_dir}/{property}_{args.grammar}_{args.target}.png')


def compare_statistics(df_1, df_2, property):
    """
    Compare the statistics of the generated and original molecules
    """
    print(f'{property} Mean generated molecules:', df_1[property].mean())
    print(f'{property} STD generated molecules:', df_1[property].std())
    print(f'{property} Mean original molecules:', df_2[property].mean())
    print(f'{property} STD original molecules:', df_2[property].std())

    
def mol_from_smiles(smiles):
    """
    This function takes a list of smiles and returns a list of molecules
    """
    
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mols = [m for m in mols if m is not None]
    return mols


def smiles_from_mol(mols):
    """
    This function takes a list of molecules and returns a list of smiles
    """
    smiles = [Chem.MolToSmiles(m, canonical=True) for m in mols]
    
    return smiles


def validity(smiles_trial):
    """
    This function takes a list of smiles and returns a list of valid smiles
    """
    mol_trial = [Chem.MolFromSmiles(s) for s in smiles_trial]
    if len(mol_trial) !=0:
        print('Fraction of valid smiles: ', 1 - mol_trial.count(None)/len(mol_trial))
        mol_trial = [m for m in mol_trial if m is not None]
        smiles_valid = smiles_from_mol(mol_trial)

        perc_valid = len(smiles_valid)/len(smiles_trial)*100
        print('Number of valid smiles: ', len(smiles_valid))
        print('Percentage of valid smiles: ', perc_valid)

        return smiles_valid, perc_valid
    else:
        print('No valid smiles')
        return None, None
    

def fraction_unique(gen, k=None, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            warnings.warn(
                "Can't compute unique@{}.".format(k) +
                "gen contains only {} molecules".format(len(gen))
            )
        gen = gen[:k]
    canonic = list(set(gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    
    print('Number of unique smiles: ', len(canonic))
    print('Percentage of unique smiles: ', len(canonic)/len(gen)*100)
    return canonic, len(canonic)/len(gen)*100

def novelty(smiles_unique, smiles_original):
    """
    This function takes a list of unique smiles and a list of original smiles and returns a list of novel smiles
    """

    smiles_novel = list(set(smiles_unique) - set(smiles_original))
    perc_novel = len(smiles_novel)/len(smiles_unique)*100

    print('Number of novel smiles: ', len(smiles_novel))
    print('Percentage of novel smiles: ', perc_novel)

    return smiles_novel, perc_novel


def logP(mol):
    """
    Computes RDKit's logP
    """
    return Chem.Crippen.MolLogP(mol)


def SA(mol):
    """
    Computes RDKit's Synthetic Accessibility score
    """
    return sascorer.calculateScore(mol)


def QED(mol):
    """
    Computes RDKit's QED score
    """
    return qed(mol)


def weight(mol):
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    return Descriptors.MolWt(mol)


def ring_size(mol):
    ri = mol.GetRingInfo()
    largest_ring_size=max((len(r) for r in ri.AtomRings()), default=0)
    return(largest_ring_size)

def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()


def fragmenter(smiles_or_mol):
    """
    fragment mol using BRICS and return smiles list
    """
    fgs = AllChem.FragmentOnBRICSBonds(smiles_or_mol)
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi


def compute_scaffold(mol, min_rings=2):
    """
    Computes scaffold for given molecule.
    Returns scaffold smiles
    """

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles

def fingerprint(mol, fp_type='morgan', radius=2, n_bits=1024):
    """
    Computes fingerprint for given molecule.
    Returns numpy array of fingerprint bits
    """
    if mol is None:
        return None
    if fp_type == 'morgan':
        fp = np.asarray(Morgan(mol, radius, nBits=n_bits))
    elif fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(mol)
        keys = np.array(keys.GetOnBits())   
        fp = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fp[keys - 1] = 1
    else:
        raise ValueError('Unknown fingerprint type')
    return fp

def fingerprints(smiles, fp_type='morgan', radius=2, n_bits=1024):
    """
    Computes fingerprints for given list of smiles.
    Returns numpy array of fingerprints n_smiles x n_bits
    """
    mols = mol_from_smiles(smiles)
    
    fps = [fingerprint(m, fp_type, radius, n_bits) for m in mols]
    fps = np.asarray(fps)
    return fps

import numpy as np

def average_agg_tanimoto(stock_vecs, gen_vecs, batch_size=5000, agg='max', p=1):
    """
    Computes average aggregated Tanimoto similarity between two sets of fingerprints
    Parameters:
        stock_vecs: numpy array of fingerprints
        gen_vecs: numpy array of fingerprints
        batch_size: size of the batch for computation
        agg: aggregation function, can be 'max' or 'mean'
        p: power for the Tanimoto similarity
    Returns:
        float
    """

    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = stock_vecs[j:j + batch_size].astype(np.float32)
        
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = gen_vecs[i:i + batch_size].astype(np.float32)
            y_gen = y_gen.T
            tp = np.dot(x_stock, y_gen)
            sum_x = np.sum(x_stock, axis=1, keepdims=True)
            sum_y = np.sum(y_gen, axis=0, keepdims=True)
            jac = tp / (sum_x + sum_y - tp)
            jac[np.isnan(jac)] = 1
            
            if p != 1:
                jac = jac ** p
                
            if agg == 'max':
                max_jac = np.max(jac, axis=0)
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(agg_tanimoto[i:i + y_gen.shape[1]], max_jac)
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += np.sum(jac, axis=0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
                
    if agg == 'mean':
        agg_tanimoto /= total
        
    if p != 1:
        agg_tanimoto = agg_tanimoto ** (1/p)
        
    return np.mean(agg_tanimoto)


def internal_diversity(smiles, fp_type='morgan', radius=2, n_bits=1024):
    """
    Computes internal diversity as: 1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if len(smiles) == 0:
        return None
    fps = fingerprints(smiles, fp_type, radius, n_bits)
    return 1 - (average_agg_tanimoto(fps, fps, agg='mean')).mean()


# %%