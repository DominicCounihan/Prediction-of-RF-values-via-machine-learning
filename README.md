   Prediction of RF, RT times (retention times) using both multilayer-perceptron and dynamic neural network models. A vector representation of each molecule from smiles is generated is Mol2vec which the model is trained on along with other features such as tpsa, molecular weight and number of rotatable bonds. RF and RT Data is uploaded to model in the form of csv files and contains experimental RF and RT times gathered from sources as cited below. 

   Requirements
   tensorflow, keras, numpy, Rdkit, pandas, os, 

   Data sources

   1 https://github.com/woshixuhao/ML-Prediction-for-Rf-values/blob/main/TLC_dataset.xlsx

   2 https://github.com/SargolMazraedoost/RT-TR/blob/main/data/dataset
