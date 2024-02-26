import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# get protein sequence
protein_fasta = open('rcsb_pdb_6P9E.fasta', 'r')
for line in protein_fasta:
    if line.startswith('>'):
        continue
    else:
        protein_seq = line.strip()
        
# join all docking results from 50k enamine predictions
dockings = glob.glob('IL36g_docking_*.csv')
dockings.sort()
df_docking = pd.concat((pd.read_csv(f) for f in dockings), ignore_index=True)
df_docking = df_docking[['ID', 'SMILES', 'glide_gscore']]
df_docking = df_docking[df_docking['glide_gscore'] != 10000] # remove failed dockings

# get docking thresholds for active (5%) and inactive compounds
gscores = df_docking['glide_gscore'].to_numpy()
threshold = np.percentile(gscores, 5)
print('Threshold for active compounds:', threshold)
print(len(df_docking[df_docking['glide_gscore'] < threshold]), 'active compounds')
print(len(df_docking[df_docking['glide_gscore'] >= threshold]), 'inactive compounds')

# plot distribution of docking scores
plt.hist(gscores, bins=100)
plt.axvline(x=threshold, color='r', linestyle='--')
plt.xlabel('Glide gscore')
plt.savefig('IL36g_docking_scores.png')

# Transform the docking scores into binary labels
def binary_label(x):
    return 0 if x >= threshold else 1

df_docking['Label'] = df_docking['glide_gscore'].apply(binary_label)
print(df_docking)
print(df_docking['Label'].value_counts())

# save the dataframe with relevant columns
df_retro = df_docking[['ID', 'SMILES', 'Label']]
df_retro['Target_ID'] = '6P9E'
df_retro['Target_seq'] = protein_seq
df_retro.rename(columns={'ID':'Compound_ID'}, inplace=True)
df_retro = df_retro[['Target_ID', 'Target_seq', 'Compound_ID', 'SMILES', 'Label']]
df_retro.to_csv('IL36g_retrospective.csv', index=False)
print(df_retro)