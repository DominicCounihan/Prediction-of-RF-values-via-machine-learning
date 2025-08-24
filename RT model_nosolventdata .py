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
from tensorflow.keras.models import Model, load_model
import joblib
# Load and prepare data

print("Loading data...")
df = pd.read_csv('RT_DATA_2.csv', 
                 names=['smiles','RT'])
df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
df['mol'] = df['mol'].apply(Chem.AddHs)


print("Calculating molecular descriptors...")
df['num_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())
df['num_heavy_atoms'] = df['mol'].apply(lambda x: x.GetNumHeavyAtoms())

def add_atom_counts(df, atom_list):
    for atom in atom_list:
        df[f'num_{atom}_atoms'] = df['mol'].apply(
            lambda x: len(x.GetSubstructMatches(Chem.MolFromSmiles(atom))))
        
add_atom_counts(df, ['C','O','N','Cl','P','Br','F','S'])

df['tpsa'] = df['mol'].apply(Descriptors.TPSA)
df['mol_wt'] = df['mol'].apply(Descriptors.ExactMolWt)
df['num_valence_electrons'] = df['mol'].apply(Descriptors.NumValenceElectrons)
#df['num_heteroatoms'] = df['mol'].apply(Descriptors.NumHeteroatoms)
#df['num_rotatablebonds']=df['mol'].apply(Descriptors.NumRotatableBonds)
#df['num_HDonors']=df['mol'].apply(Descriptors.NumHDonors)
#df['num_HAcceptors']=df['mol'].apply(Descriptors.NumHAcceptors)
#df['MolLogP']=df['mol'].apply(Descriptors.MolLogP)
 

# Download and load Mol2Vec model
print("Setting up Mol2Vec model...")
model_path = 'model_300dim.pkl'
if not os.path.exists(model_path):
    print("Downloading Mol2Vec model...")
    url = 'https://github.com/samoturk/mol2vec_notebooks/raw/master/Notebooks/model_300dim.pkl'
    r = requests.get(url, allow_redirects=True)
    open(model_path, 'wb').write(r.content)
    
w2v_model = word2vec.Word2Vec.load(model_path)

# Generate Mol2Vec embeddings with proper dimensionality
print("Generating Mol2Vec embeddings...")
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

df['mol2vec'] = get_mol2vec_embeddings(df['mol'], w2v_model)



# Skip first sample in all data to match solvent_features
df = df.iloc[1:].reset_index(drop=True)
print(f"Adjusted dataframe shape: {df.shape}")
# Prepare data for model
y = df['RT'].values
X_mol2vec = [x for x in df['mol2vec']]  # List of (n_tokens, 300) arrays
X_other = df.drop(['smiles','mol','RT','mol2vec'], axis=1).values

# Normalize features
scaler = StandardScaler()
X_other = scaler.fit_transform(X_other)

# Train/val/test split
X_train_vec, X_temp_vec, y_train, y_temp = train_test_split(
    X_mol2vec, y, test_size=0.3, random_state=42)
X_val_vec, X_test_vec, y_val, y_test = train_test_split(
    X_temp_vec, y_temp, test_size=0.5, random_state=42)

X_train_other, X_temp_other, _, _ = train_test_split(
    X_other, y, test_size=0.3, random_state=42)
X_val_other, X_test_other, _, _ = train_test_split(
    X_temp_other, y_temp, test_size=0.5, random_state=42)

# Corrected sequence preparation function
def prepare_sequences(mol2vec_list, other_features, max_len=100, vec_dim=100):
    """Convert mol2vec embeddings to fixed-length sequences"""
    sequences = []
    
    for vec in mol2vec_list:
        # Ensure proper shape (n_tokens, 300)
        if len(vec.shape) == 1:
            vec = vec.reshape(1, -1)
        
        # Verify dimensions
        if vec.shape[1] != vec_dim:
            raise ValueError(f"Expected embedding dimension {vec_dim}, got {vec.shape[1]}")
        
        # Pad/truncate to max_len
        if vec.shape[0] > max_len:
            seq = vec[:max_len]
        else:
            seq = np.zeros((max_len, vec_dim))
            seq[:vec.shape[0], :] = vec  # Corrected this line
        
        sequences.append(seq)
    
    sequences = np.array(sequences)
    
    # Prepare other features to match dimension
    other_features = np.array(other_features)
    other_features = other_features.reshape(-1, 1, other_features.shape[1])
    other_padded = np.zeros((other_features.shape[0], 1, vec_dim))
    other_padded[:, :, :other_features.shape[2]] = other_features
    
    return np.concatenate([other_padded, sequences], axis=1)

print("Preparing sequences...")
X_train = prepare_sequences(X_train_vec, X_train_other, vec_dim=100)
X_val = prepare_sequences(X_val_vec, X_val_other, vec_dim=100)
X_test = prepare_sequences(X_test_vec, X_test_other, vec_dim=100)

# Build model
print("Building model...")
def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Process molecular features (position 0)
    mol_features = layers.Flatten()(inputs[:, 0:1, :])
    
    # Process sequence embeddings (positions 1-100)
    seq = inputs[:, 1:, :]
    
    # Parallel convolution pathways
    conv1 = layers.Conv1D(1700, 2, activation='relu')(seq)
    pool1 = layers.GlobalMaxPooling1D()(conv1)
    
    conv2 = layers.Conv1D(500, 1, activation='relu')(seq)
    pool2 = layers.GlobalMaxPooling1D()(conv2)
    pool12 = layers.concatenate([pool1,pool2])
    fcoutput1  = (layers.Dense(1000,activation="relu"))(pool12)
    fcoutput1  = (layers.Dense(600,activation="relu"))(fcoutput1)
    fcoutput1  = (layers.Dense(300,activation="relu"))(fcoutput1)
    istm = layers.Bidirectional(layers.LSTM(500, return_sequences=False,implementation=1,name="istm_1"))(seq)
    d1 = (layers.Dense(500,activation="relu"))(istm)
    d1 = (layers.Dense(700, activation="relu"))(d1)
    fcoutput2 = (layers.Dense(200,activation="relu"))(d1)
    fc = layers.concatenate([fcoutput1, fcoutput2])
    fcoutput = (layers.Dense(1,activation="relu"))(fc)
    
    # Combine all features
    combined = layers.concatenate([mol_features,pool12,fcoutput])
    dense = layers.Dense(128, activation='relu')(combined)
    output = layers.Dense(1)(dense)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


model = build_model((101, 100))
model.summary()

# Train model
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=100,
    verbose=2,shuffle=True, validation_split=0.11
)

# Evaluate
print("Evaluating model...")
test_results = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_results[0]:.4f}, Test MAE: {test_results[1]:.4f}")

# Make predictions
y_pred = model.predict(X_test).flatten()

# Calculate additional metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.4f}")
joblib.dump(scaler,'scaler.save')
# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training History')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('RT value')
plt.ylabel('Predicted RT')
plt.title('Prediction Performance')

plt.tight_layout()
plt.show()

joblib.dump(scaler,'scaler_RT.save')
model.save('model_RT_nosolvent.keras')

def predict_rt(smiles):

    # Load the model
    if os.path.exists('model_RT_nosolvent.keras'):
        model = tf.keras.models.load_model('model_RT_nosolvent.keras')
    else:
        raise FileNotFoundError("Model not found. Please train the model first.")
    
    # Load the scaler
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
        'num_valence_electrons'
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
    
    mol_desc_values = [mol_desc[name] for name in descriptor_names]
    
    # Scale features
    features_scaled = scaler.transform(np.array(mol_desc_values).reshape(1, -1))
    
    # Generate Mol2Vec embedding
    mol2vec_embedding = get_mol2vec_embeddings([mol], w2v_model)[0]
    
    # Prepare sequence
    max_len = 100
    vec_dim = 100
    
    if len(mol2vec_embedding.shape) == 1:
        mol2vec_embedding = mol2vec_embedding.reshape(1, -1)
    
    if mol2vec_embedding.shape[0] > max_len:
        seq = mol2vec_embedding[:max_len]
    else:
        seq = np.zeros((max_len, vec_dim))
        seq[:mol2vec_embedding.shape[0], :] = mol2vec_embedding
    
    # Prepare other features to match dimension
    other_features = features_scaled.reshape(-1, 1, features_scaled.shape[1])
    other_padded = np.zeros((other_features.shape[0], 1, vec_dim))
    other_padded[:, :, :other_features.shape[2]] = other_features
    
    # Combine into final input sequence
    final_input = np.concatenate([other_padded, seq.reshape(1, max_len, vec_dim)], axis=1)
    
    # Make prediction
    return model.predict(final_input)[0][0]

# Test the function with a sample SMILES
test_smiles = 'Cn1cnc2c1c(=O)n(C)c(=O)n2C'  # caffeine
try:
    rt_prediction = predict_rt(test_smiles)
    print(f"Predicted RT for {test_smiles}: {rt_prediction:.4f}")
except Exception as e:
    print(f"Error predicting RT: {e}")
