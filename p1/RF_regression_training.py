# This trains and tests a random forest regressor on all of the data
# It outputs a csv with all of the predictions to be uploaded to Kaggle

import random
import pandas as pd
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol as fMol


def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")


print "Loading training data..."
with open('mol_data/all_mols.mol') as f:
    train_mols = pickle.load(f)
with open('mol_data/train_gap.pkl') as f:
    train_gap = pickle.load(f)


print "Generating training morgan fingerprints..."
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in train_mols]

np_fps = []
for fp in fps:
  arr = np.zeros((1,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_fps.append(arr)


print "Fitting model..."
RF = RandomForestRegressor()
RF.fit(np_fps, train_gap)


print "Saving model..."
with open('mol_data/RF_Regressor_all_data.pkl') as f:
    pickle.dump(RF, f)


print "Loading test data..."
with open('mol_data/test_mols.mol') as f:
    test_mols = pickle.load(f)


print "Generating test morgan fingerprints..."
test_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in test_mols]

test_np_fps = []
for fp in test_fps:
  arr = np.zeros((1,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  test_np_fps.append(arr)


print "Predicting test data..."
RF_test_pred = RF.predict(test_np_fps)


print "Saving test predictions"
write_to_file('RF_all_train_test_predictions.csv', RF_test_pred)
