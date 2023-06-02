# -*- coding: utf-8 -*-
"""
@Time    : 24/05/2023 11:28
@Author  : dorianhgn
@FileName: custom_scaler.py
@Software: PyCharm
Je créée ici un StandardScaler + une tanh
"""

import numpy as np
import pandas as pd

mean_std = pd.read_csv('data/mean_std.csv', sep=';').set_index('Unnamed: 0')


class CustomScaler:
    def __init__(self):
        self.mean = mean_std['0']
        self.std = mean_std['1']

    """def fit(self, X):
        # Calculez les statistiques de référence sur l'ensemble d'entraînement complet
        self.reference_stats = {'mean': np.mean(X), 'std': np.std(X)}""" # Pas besoin ici

    def transform(self, X):
        # Utilisez les statistiques de référence pour effectuer la transformation
        scaled_X = np.tanh((X - self.mean) / self.std)
        return scaled_X.to_numpy()

    def inverse_transform(self, scaled_X):
        # Utilisez les statistiques de référence pour inverser la transformation
        X = np.arctanh(scaled_X * self.std + self.mean)
        return X