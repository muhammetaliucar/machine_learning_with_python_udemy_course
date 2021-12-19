# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:00:43 2021

@author: Aliucar
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

#veriyi okuduk.
data = pd.read_csv("eksikveriler.csv")
#veriden yaş kolonunu çektik.
yas = data.iloc[:,1:4].values
#imputer ile öğrettik.
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
#imputer ile uyguladık
data.iloc[:,1:4] = imputer.fit_transform(data.iloc[:,1:4])

#ulke kolonunu alıyoruz.
#amaç ülke kolonunu sayısal bir değişkene dönüştürmek
ulke = data.iloc[:,0:1].values

#gerekli kütüphaneyi import ediyoruz.
from sklearn import preprocessing

#label encoder kısaltması için:le. label=etiket,encoder=kodlayıcı
le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(data.iloc[:,0])

#böylelikle tr ülkeleri sayısal bir karşılığa çevirdik.

#şimdi 1-0-0 lı hale getireceğiz.

ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()
