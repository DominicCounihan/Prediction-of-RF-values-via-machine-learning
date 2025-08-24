import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from mol2vec.features import mol2alt_sentence, MolSentence
from gensim.models import word2vec
import requests
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers
import csv
import joblib

model_path = 'model_300dim.pkl'
if not os.path.exists(model_path):
    print("Downloading Mol2Vec model...")
    url = 'https://github.com/samoturk/mol2vec_notebooks/raw/master/Notebooks/model_300dim.pkl'
    r = requests.get(url, allow_redirects=True)
    open(model_path, 'wb').write(r.content)
    
w2v_model = word2vec.Word2Vec.load(model_path)

def get_mol2vec_embeddings(mols, model, radius=1, unseen='UNK'):
    """Generate properly dimensioned mol2vec embeddings"""
    sentences = [MolSentence(mol2alt_sentence(mol, radius)) for mol in mols]
    keys = set(model.wv.key_to_index.keys())
    
    if unseen:
        unseen_vec = model.wv.get_vector(unseen)
    
    embeddings = []
    for sentence in sentences:
        vecs = []
        for token in sentence:
            if token in keys:
                vecs.append(model.wv.get_vector(token))
            elif unseen:
                vecs.append(unseen_vec)
        
        if len(vecs) == 0:
            # If no tokens matched, use zero vector
            vecs.append(np.zeros(model.vector_size))
            
        # Stack tokens to create (n_tokens, 300) array
        mol_vec = np.vstack(vecs)
        embeddings.append(mol_vec)
    
    return embeddings

smiles='Cn1cnc2c1c(=O)n(C)c(=O)n2C'
eluent_percents=[0,0,0,0,1]

if os.path.exists('model_RF_5.keras'):
        model = tf.keras.models.load_model('model_RF_5.keras')
else:
        raise FileNotFoundError("Model not found. Please train the model first.")
    
if os.path.exists('scaler.save'):
        scaler = joblib.load('scaler.save')

else:
        raise FileNotFoundError("Scaler not found. Please train the model first to generate the scaler.")
    
    # Process input molecule
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    raise ValueError("Invalid SMILES string")
mol = Chem.AddHs(mol)
    
# Compute molecular descriptors (same as training)
descriptor_names = [
        'num_atoms', 'num_heavy_atoms', 'num_C_atoms', 'num_O_atoms', 
        'num_N_atoms', 'num_Cl_atoms', 'num_P_atoms', 'num_Br_atoms', 
        'num_F_atoms', 'num_S_atoms', 'tpsa', 'mol_wt', 
        'num_valence_electrons', 'num_heteroatoms', 'num_rotatablebonds',
        'num_HDonors', 'num_HAcceptors', 'MolLogP'
    ]
    
mol_desc = {}
mol_desc['num_atoms'] = mol.GetNumAtoms()
mol_desc['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
    
atom_list = ['C','O','N','Cl','P','Br','F','S']
for atom in atom_list:
        key = f'num_{atom}_atoms'
        mol_desc[key] = len(mol.GetSubstructMatches(Chem.MolFromSmiles(atom)))
    
mol_desc['tpsa'] = Descriptors.TPSA(mol)
mol_desc['mol_wt'] = Descriptors.ExactMolWt(mol)
mol_desc['num_valence_electrons'] = Descriptors.NumValenceElectrons(mol)
mol_desc['num_heteroatoms'] = Descriptors.NumHeteroatoms(mol)
mol_desc['num_rotatablebonds'] = Descriptors.NumRotatableBonds(mol)
mol_desc['num_HDonors'] = Descriptors.NumHDonors(mol)
mol_desc['num_HAcceptors'] = Descriptors.NumHAcceptors(mol)
mol_desc['MolLogP'] = Descriptors.MolLogP(mol)
    
mol_desc_values = [mol_desc[name] for name in descriptor_names]
    
# Process solvent properties (6 features)
eluents = ['CCCCCC','O=C(OCC)C','ClCCl','CO','CCOCC']
all_properties = []
for smi in eluents:
        mol_e = Chem.MolFromSmiles(smi)
        properties = [
            Descriptors.TPSA(mol_e),
            Descriptors.ExactMolWt(mol_e),
            Descriptors.NumHDonors(mol_e),
            Descriptors.NumHAcceptors(mol_e),
            Descriptors.NumRotatableBonds(mol_e),
            Descriptors.MolLogP(mol_e)
        ]
all_properties.append(properties)

matrix_properties = np.array(all_properties).T
    
solvent_features = np.sum(matrix_properties * np.array(eluent_percents), axis=1)
    
all_features = np.concatenate([mol_desc_values, solvent_features])
    
features_scaled = scaler.transform(all_features.reshape(1, -1))

mol2vec_embedding = get_mol2vec_embeddings([mol], w2v_model)[0]

max_len = 100
vec_dim = 100
    
if len(mol2vec_embedding.shape) == 1:
        mol2vec_embedding = mol2vec_embedding.reshape(1, -1)
    
if mol2vec_embedding.shape[0] > max_len:
        seq = mol2vec_embedding[:max_len]
else:
        seq = np.zeros((max_len, vec_dim))
        seq[:mol2vec_embedding.shape[0], :] = mol2vec_embedding
    
other_features = features_scaled.reshape(-1, 1, features_scaled.shape[1])
other_padded = np.zeros((other_features.shape[0], 1, vec_dim))
other_padded[:, :, :other_features.shape[2]] = other_features
    
final_input = np.concatenate([other_padded, seq.reshape(1, max_len, vec_dim)], axis=1)
    
rf_value=model.predict(final_input)[0][0]

print(f"Predicted RF: {rf_value:.4f}")
      
