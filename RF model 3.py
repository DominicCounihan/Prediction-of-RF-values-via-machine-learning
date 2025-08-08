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
# Load and prepare data

print("Loading data...")
df = pd.read_csv('RF_DATA.csv', 
                 names=['smiles', 'H','EA','DCM','MeOH','Et2O','RF'])
print(df['H'])
df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
df['mol'] = df['mol'].apply(Chem.AddHs)
file_rf_rows= open('Rf_Data.csv')

eluients = ['CCCCCC','O=C(OCC)C','ClCCl','CO','CCOCC']


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
df['num_heteroatoms'] = df['mol'].apply(Descriptors.NumHeteroatoms)
df['num_rotatablebonds']=df['mol'].apply(Descriptors.NumRotatableBonds)
df['num_HDonors']=df['mol'].apply(Descriptors.NumHDonors)
df['num_HAcceptors']=df['mol'].apply(Descriptors.NumHAcceptors)
df['MolLogP']=df['mol'].apply(Descriptors.MolLogP)
 
def process_eluient():
    
     # Load data
    eluents = ['CCCCCC','O=C(OCC)C','ClCCl','CO','CCOCC']
    df = pd.read_csv('RF_DATA.csv', names=['smiles', 'H', 'EA', 'DCM', 'MeOH', 'Et2O', 'RF'])

    # Calculate molecular properties for each eluent
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

    # Create property matrix (5 eluents × 6 properties)
    matrix_properties = np.array(all_properties)
    print("Property matrix shape:", matrix_properties.shape)

    # Convert el_data to numeric, coercing errors to NaN
    el_data = np.array([
        pd.to_numeric(df['H'], errors='coerce').values,
        pd.to_numeric(df['EA'], errors='coerce').values,
        pd.to_numeric(df['DCM'], errors='coerce').values,
        pd.to_numeric(df['MeOH'], errors='coerce').values,
        pd.to_numeric(df['Et2O'], errors='coerce').values
    ])

    # Remove first column if needed
    el_data = np.delete(el_data, 0, axis=1)
    el_data = np.nan_to_num(el_data)  # Converts NaNs to 0

    # Transpose property matrix to (6 properties × 5 eluents)
    matrix_properties = matrix_properties.T
    print("Transposed property matrix shape:", matrix_properties.shape)

    # Initialize result arrays as 2D arrays with proper dimensions
    num_samples = el_data.shape[1]
    el_TPSA = np.zeros((1, num_samples))  # Initialize as 2D row vector
    el_Exact_MolWt = np.zeros((1, num_samples))
    el_NumHDonors = np.zeros((1, num_samples))
    el_NumHAcceptors = np.zeros((1, num_samples))
    el_NumRotBonds = np.zeros((1, num_samples))
    el_MolLogP = np.zeros((1, num_samples))

    # Calculate weighted properties
    for prop in range(5):  # For each eluent (0 to 4)
        # Stack new rows to each property array
        el_TPSA = np.vstack([el_TPSA, el_data[prop, :] * matrix_properties[0, prop]])
        el_Exact_MolWt = np.vstack([el_Exact_MolWt, el_data[prop, :] * matrix_properties[1, prop]])
        el_NumHDonors = np.vstack([el_NumHDonors, el_data[prop, :] * matrix_properties[2, prop]])
        el_NumHAcceptors = np.vstack([el_NumHAcceptors, el_data[prop, :] * matrix_properties[3, prop]])
        el_NumRotBonds = np.vstack([el_NumRotBonds, el_data[prop, :] * matrix_properties[4, prop]])
        el_MolLogP = np.vstack([el_MolLogP, el_data[prop, :] * matrix_properties[5, prop]])

    # Sum along the rows (axis=0) to get 1D arrays
    el_TPSA_sum = np.sum(el_TPSA, axis=0)
    el_Exact_MolWt_sum = np.sum(el_Exact_MolWt, axis=0)
    el_NumHDonors_sum = np.sum(el_NumHDonors, axis=0)
    el_NumHAcceptors_sum = np.sum(el_NumHAcceptors, axis=0)
    el_NumRotBonds_sum = np.sum(el_NumRotBonds, axis=0)
    el_MolLogP_sum = np.sum(el_MolLogP, axis=0)

    # Combine all properties into final matrix
    print(el_TPSA_sum)
    print(el_Exact_MolWt_sum)

    el_data_props = np.vstack([
        el_TPSA_sum, 
        el_Exact_MolWt_sum, 
        el_NumHDonors_sum,
        el_NumHAcceptors_sum,
        el_NumRotBonds_sum,
        el_MolLogP_sum
    ])


    
    
        #work out what solvent corresponds to what row
        #get ratios each solvent
        #work out each weighted properties by multiplying by vector
        #average them

    return el_data_props

el_data_props= process_eluient()

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

el_data_props = process_eluient()
solvent_features = el_data_props.T

# Skip first sample in all data to match solvent_features
df = df.iloc[1:].reset_index(drop=True)
print(f"Adjusted dataframe shape: {df.shape}")
# Prepare data for model
y = df['RF'].values
X_mol2vec = [x for x in df['mol2vec']]  # List of (n_tokens, 300) arrays
X_other = df.drop(['smiles','mol','RF','mol2vec'], axis=1).values

X_other = np.concatenate([X_other, solvent_features], axis=1)
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
    conv1 = layers.Conv1D(256, 3, activation='relu')(seq)
    pool1 = layers.GlobalMaxPooling1D()(conv1)
    
    conv2 = layers.Conv1D(256, 5, activation='relu')(seq)
    
    pool2 = layers.GlobalMaxPooling1D()(conv2)
    
    conv3 = layers.Conv1D(256, 7, activation='relu')(seq)

    pool3 = layers.GlobalMaxPooling1D()(conv3)

    # Combine all features
    combined = layers.concatenate([mol_features, pool1, pool2, pool3])
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
    epochs=20,
    batch_size=64,
    verbose=1
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
plt.xlabel('RF value')
plt.ylabel('Predicted RF')
plt.title('Prediction Performance')

plt.tight_layout()
plt.show()
