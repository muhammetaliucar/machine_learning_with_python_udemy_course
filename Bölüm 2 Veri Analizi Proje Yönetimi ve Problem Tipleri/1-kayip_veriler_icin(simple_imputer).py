# -*- coding: utf-8 -*-
"""
Created on Sun May  9 17:00:27 2021

@author: Aliucar
"""

#eksik veriler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# kayıp veriler için 
from sklearn.impute  import SimpleImputer

data = pd.read_csv("eksikveriler.csv")

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
yas = data.iloc[:,1:4].values

imputer = imputer.fit(yas[:,1:4])

yas[:,1:4] = imputer.transform(yas[:,1:4])