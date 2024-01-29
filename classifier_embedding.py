import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
import torch
import esm
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import selfies as sf

# LOAD DATASET
print('DATASET:')
df = pd.read_csv('dude_full.tsv', sep='\t')
print("Shape of dataframe: ", df.shape)
#print(df.head())

# Check data
print("Number of proteins: ", len(set(df['Target_ID'].tolist())))
print("Number of active compounds: ", len(df[df['Label'] == 1]))
print("Number of inactive compounds (decoys): ", len(df[df['Label'] == 0]))

# Here, we sample the df because of the time for the test
# df = df.sample(n=5)

######## MOLECULE ENCODING ########
def smiles_to_fp(smiles, method='Morgan4', n_bits=2048):
    """
    Encode a molecule from a SMILES string into a fingerprint.

    Parameters
    ----------
    smiles : str
        The SMILES string defining the molecule.
    method : str
        The type of fingerprint to use. Default is Morgan4.
    n_bits : int
        The length of the fingerprint.

    Returns
    -------
    array
        The fingerprint array.
    """
    
    mol = Chem.MolFromSmiles(smiles)
    
    if method == 'Morgan4':
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=n_bits)
        fpg_array = np.array(fpg.GetFingerprint(mol))
        return fpg_array
        #fpg_tensor = torch.from_numpy(fpg_array)
        #fpg_tensor.to('cuda')
        #return fpg_tensor
    else:
        print("Sorry, but the other methods are not implemented...")
    
# compound_df = df.copy()
# compound_df['fp'] = compound_df['Molecule_SMILES'].apply(smiles_to_fp)

def smiles_to_molformer():
    """
    Encode a molecule from a SMILES string into a MolFormer-XL embedding.
    
    From some reasons, this function is not working in the current directory,
    but it works in the MolFormer directory. So, I will use it from there.
    Problems with the import of the MolFormer-XL model.
    
    This model should be run using the conda environment: MolTran_CUDA11
    """
    
    os.system('/data2/julia/PLM/MolFormer/molformer/notebooks/pretrained_molformer/embed_dude.py')
    
    # it stores the embeddings in the file: ESM2650M-MolFormer/compound_df.pkl
    
def smiles_to_chemberta(smiles, model_version='v2'):
    """_summary_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # Load the model
    if model_version == 'v1':
        model_name = 'seyonec/ChemBERTa-zinc-base-v1'
    elif model_version == 'v2':
        model_name = 'DeepChem/ChemBERTa-77M-MLM' #ChemBERTA-2
    else:
        raise ValueError('Model version must be either v1 or v2')
    
    model = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True).to('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Embed the smiles
    input_token = tokenizer(smiles, return_tensors='pt').to('cuda:0')
    output = model(**input_token, output_hidden_states=True)
    embedding = output.hidden_states[0]
    embedding = embedding.cpu().detach().numpy()
    #print(embedding.shape)
    embedding = np.squeeze(embedding)
    embedding = embedding.mean(axis=0)
    print(embedding.shape)
    return embedding

def smiles_to_selformer(smiles):
    """_summary_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    # Load the model
    model_name = '/data/ifilella/repos/SELFormer/models/modelC/'
    config = RobertaConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    tokenizer = RobertaTokenizer.from_pretrained('/data/ifilella/repos/SELFormer/data/RobertaFastTokenizer')
    model = RobertaModel.from_pretrained(model_name, config=config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model.to(device)
    
    # SMILE to SELFIE
    selfies = sf.encoder(smiles)
    print(selfies)
    
    # Embed the smiles
    token = torch.tensor([tokenizer.encode(selfies, add_special_tokens=True, max_length=512, padding=True, truncation=True)]).to(device)
    output = model(token, output_hidden_states=True)
    sequence_out = output.hidden_states[0]
    shape = sequence_out.cpu().detach().numpy().shape
    embedding = torch.mean(sequence_out[0], dim=0).cpu().detach().numpy()
    shape = embedding.shape
    print(shape)
    return embedding


######## PROTEIN ENCODING ########
def protseqs_to_esm(prots_df, col_seq, col_id, esm_model = '650M'):
    """
    Encode a protein sequence string into a ESM model embedding.

    Parameters
    ----------
    seqs : str
        The protein aa sequence string.
    ids : str
        The protein identifiers.
    esm_model : str
        The type of ESM model to use. It can be 650M or 3B.
    
    Returns
    -------
    array
        The protein embedding array.
    """
    # Select GPU of CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load ESM-2 Model
    if esm_model == '650M':
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        layers = 33
    elif esm_model == '3B':
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        layers = 36
    else:
        raise ValueError('Model must be either 650M or 3B')
    
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    # Get the sequence embeddings
    seqs_df = prots_df[[col_seq, col_id]]
    seqs_df = seqs_df.drop_duplicates()
    
    for index,row in seqs_df.iterrows():
        data = [(row[col_id], row[col_seq])]
        print(data)
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        # per-residue embeddings
        with torch.no_grad():
            batch_tokens = batch_tokens.to(device)
            results = model(batch_tokens, repr_layers=[layers], return_contacts=False)
        token_embeddings = results['representations'][layers]
    
        # per-sequence embeddings
        sequence_embeddings = token_embeddings[0, 1:len(row[col_seq]) + 1].mean(0)
        sequence_embeddings = sequence_embeddings.cpu()
        sequence_embeddings = sequence_embeddings.numpy()
        row[col_seq] = sequence_embeddings
    seqs_df = seqs_df.rename({'Target_Seq':'esm'}, axis=1)
    
    df_return = pd.merge(prots_df, seqs_df, on='Target_ID')
    return df_return
    
# new_compound_df = protseqs_to_esm(prots_df=compound_df, col_seq='Target_Seq', col_id='Target_ID', esm_model='3B')


######## COEMBEDDING ########
def coembedding(prot_emb, comp_emb):
    prot_tensor = torch.from_numpy(prot_emb)
    comp_tensor = torch.from_numpy(comp_emb)
    prot_tensor = prot_tensor.to('cuda')
    comp_tensor = comp_tensor.to('cuda')
    co = torch.cat((prot_tensor, comp_tensor), 0)
    co = co.cpu()
    co = co.numpy()
    return co
# new_compound_df['coembed'] = new_compound_df.apply(lambda x: coembedding(x['esm'], x['fp']), axis = 1)


######## EXECUTE THE FUNCTIONS HERE ########
# COMBINE DIFFERENT EMBEDDINGS AND SAVE THEM
if __name__ == '__main__':
    
    compound_df = df.copy()
    
    # Embed molecules
    #print('\nEmbedding SMILES to fingerprints...\n')
    #compound_df['fp'] = compound_df['Molecule_SMILES'].apply(smiles_to_fp)
    
    # print('\nEmbedding SMILES to MolFormer-XL embedding...\n')
    
    # print('\nEmbedding SMILES to ChemBERTa embedding v1...\n')
    # compound_df['chemberta'] = compound_df['Molecule_SMILES'].apply(smiles_to_chemberta, model_version='v1')
    # compound_df.to_pickle('ESM2650M-ChemBerta/compound_df.pkl')
    
    # print('\nEmbedding SMILES to ChemBERTa embedding v2...\n')
    # compound_df['chembert2'] = compound_df['Molecule_SMILES'].apply(smiles_to_chemberta, model_version='v2')
    # compound_df.to_pickle('ESM2650M-ChemBERT2/compound_df.pkl')
    
    # print('\nEmbedding SMILES to SELFormer embedding...\n')
    # compound_df['selformer'] = compound_df['Molecule_SMILES'].apply(smiles_to_selformer)
    
    
    # Embed proteins
    # print('\nEmbedding protein sequences to ESM...\n')
    # compound_df = pd.read_pickle('ESM2650M-MolFormer/compound_df.pkl')
    # new_compound_df = protseqs_to_esm(prots_df=compound_df, col_seq='Target_Seq', col_id='Target_ID', esm_model='650M')
    # print(new_compound_df)
    # new_compound_df.to_csv('ESM2650M-MolFormer/new_compound_df.csv')
    # new_compound_df.to_pickle('ESM2650M-MolFormer/new_compound_df.pkl')
    
    
    # make coembedding protein-compounds
    print('\nCoembedding proteins and compounds...\n')
    new_compound_df = pd.read_pickle('ESM2650M-MolFormer/new_compound_df.pkl')
    print(new_compound_df)
    exit()
    new_compound_df['coembed'] = new_compound_df.apply(lambda x: coembedding(x['esm'], x['selformer']), axis = 1)
    print(new_compound_df)
    
    # store embeddings
    new_compound_df.to_csv('data_embed.csv')
    new_compound_df.to_pickle('data_embed.pkl')