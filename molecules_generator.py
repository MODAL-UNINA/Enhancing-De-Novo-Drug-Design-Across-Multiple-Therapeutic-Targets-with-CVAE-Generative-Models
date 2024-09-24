#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
#Imports
import sys
import argparse
import numpy as np
import pandas as pd
import selfies as sf
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

from rdkit import Chem, RDLogger
from rdkit import RDLogger
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Descriptors import ExactMolWt, NumHAcceptors, NumHDonors, NumRotatableBonds


from utils import *
from model_cvae import load_model


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
# %%
# Set working directory
work_dir = Path(__file__)
root_dir = work_dir.parents[0]

sys.path.append(str(root_dir))


parser = argparse.ArgumentParser(description='Model')
parser.add_argument('--dataset_dir', type=str, default=f'{root_dir}/data', help='Dataset directory')
parser.add_argument('--images_dir', type=str, default=f'{root_dir}/images', help='Images directory')
parser.add_argument('--save_dir', type=str, default=f'{root_dir}/save', help='Save directory')
parser.add_argument('--gen_dir', type=str, default=f'{root_dir}/generated', help='Generated directory')
parser.add_argument('--grammar', type=str, default='smiles', help='smiles or selfies')
parser.add_argument('--model_type', type=str, default='tuning', help='initial or tuning')
parser.add_argument('--target', type=str, default='cdk2', help='cdk2, gamma, dpp4')
parser.add_argument('--num_properties', type=int, default=6, help='Number of properties')
parser.add_argument('--seq_length', type=int, default=120, help='Sequence length')
parser.add_argument('--num_iterations', type=int, default=10, help='Number of iterations')
parser.add_argument('--num_trials', type=int, default=3000, help='Number of trials')
args, _ = parser.parse_known_args()

prop_file = f'{args.dataset_dir}/{args.grammar}_{args.target}.txt'

# %%
# Load model and generate molecules
args.vocab, chars, labels = load_vocab(prop_file, args)

scaler = MinMaxScaler()
labels = scaler.fit_transform(labels)
labels = np.round(labels, 2)

model = load_model(args)

range_list2 = [(300, 500), (0, 5), (0, 50), (0, 5), (0, 5), (0, 5)]
if args.model_type == 'tuning':
    if args.target == 'cdk2':
        range_list2 = [455, 1, 38, 5, 2, 4]
    if args.target == 'gamma':
        range_list2 = [382, 4, 64, 1, 1, 9]
    if args.target == 'dpp4':
        range_list2 = [363, 2, 61, 3, 2, 7]

if args.grammar == 'selfies':
    with open(prop_file, 'r') as file:
        lines = file.readlines()
    df = pd.DataFrame([line.split() for line in lines]).drop(index=0)
        
    sf.set_semantic_constraints('hypervalent')
    smiles_original = [sf.decoder(SELFIES) for SELFIES in df.iloc[:, 0]]
else:
    smiles_original = pd.read_csv(prop_file, header=None, sep='\t')[0]

smiles_original = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles_original[1:]]
# %%
# Generate molecules
trials = []

for iter in range(args.num_iterations):
    trial = []
    C = []
    if args.model_type == 'tuning':
        C = range_list2
        C = np.array(C).T
    else:
        for i in range(len(range_list2)):
            C.append(np.random.randint(range_list2[i][0], range_list2[i][1], 1)[0])
        
    print('The conditions are:', C)

    # Normalize C
    C = np.tile(C, (args.num_trials, 1))

    C = np.round(scaler.transform(C), 2)
    start_codon = np.array([np.array(list(map(args.vocab.get, 'X')))for _ in range(args.num_trials)])
    smiles_generated = []
    x = model.generate (C, start_codon)

    # Convert to smiles usinf from utils convert_to_smiles
    for i in range(C.shape[0]):
        smiles_generated += [convert_to_smiles(x[i].numpy(), chars)]

    trial += list([s.split('E')[0] for s in smiles_generated])
    
    if args.grammar == 'selfies':
        smi = []
        try:
            sf.set_semantic_constraints('hypervalent')
            smi += [sf.decoder(m) for m in trial]
        except:
            print('Decoder Error')
    
        trial = [m for m in smi if m is not None]

    trials += trial
    print('Number of generated smiles : ', len(trial))

smiles_valid_all, percent_valid_all = validity(trials)
smiles_unique_all, perc_unique_all = fraction_unique(smiles_valid_all, check_validity=True)
smiles_novel_all, perc_novel_all = novelty(smiles_unique_all, smiles_original)

ms = [Chem.MolFromSmiles(s) for s in smiles_novel_all]
ms = [m for m in ms if m is not None]
print ('number of valid unique and novel smiles : ', len(ms))

with open(os.path.join(args.gen_dir, f'generated_molecules_{args.grammar}_{args.target}.txt'), 'w') as w:
    w.write('smiles\tMW\tLogP\tTPSA\tHBA\tHBD\tRotB\n')
    for m in ms:
        try:
            w.write('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' %(Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcTPSA(m), NumHAcceptors(m), NumHDonors(m), NumRotatableBonds(m)))
        except:
            continue  

# %%
