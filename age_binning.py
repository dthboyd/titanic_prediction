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

n_bins = []
age_lim = []
n=1000
for i in range(n):
    transformer = MDLP(random_state=i, continuous_features=None)
    age_dis = transformer.fit_transform(age, survived)
    age_bins = transformer.cat2intervals(age_dis, 0)
    n_bins.append(len(set(age_bins)))
    if len(set(age_bins))==2: age_lim.append(age_bins[0])
    elif len(set(age_bins))>2: print('\t ! more than two bins, n=', len(set(age_bins)))

print('* estimated N bins:', set(n_bins))
print('\t mean', np.mean(1.*np.array(n_bins)))
print('* Age thresholds, frequencies')
lim_val = np.array(age_lim)[:,0]

sum_not_inf = 0
for val_i in set(lim_val):
    print('\t', val_i, (1.*sum(lim_val==val_i))/n)
    sum_not_inf = sum_not_inf+sum(lim_val==val_i)
print('\t', 'inf', (n-sum_not_inf)/n)
