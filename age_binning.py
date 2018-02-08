"""
A minimal test of best binning for Age.
Uses 'Minimum Description Length Binning' method by Usama Fayyad  et al.
implemented by Henry Lin et al. at https://github.com/hlin117/mdlp-discretization

"""

import pandas as pd
import numpy as np

from mdlp.discretization import MDLP

train_raw = pd.read_csv("input/train.csv")
test_raw = pd.read_csv("input/test.csv")

# drop NaNs, use only the Age feature itself to estimate bins
train_sur_age = train_raw[['Survived','Age']].dropna(axis=0)
survived = train_sur_age['Survived'].values
age = (train_sur_age['Age'].values).reshape(-1, 1)

transformer = MDLP(random_state=666, continuous_features=None)
age_dis = transformer.fit_transform(age, survived)
age_bins = transformer.cat2intervals(age_dis, 0)

print(set(age_bins))
