# This trains a random forest regressor on all of the data

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

print "Loading data..."

with open('mol_data/all_mols.mol') as f:
    train_mols = pickle.load(f)
with open('mol_data/train_gap.pkl') as f:
    train_gap = pickle.load(f)


print "Generating morgan fingerprints..."

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