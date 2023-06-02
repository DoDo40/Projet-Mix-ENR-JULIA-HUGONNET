# -*- coding: utf-8 -*-
"""
@Time    : 22/05/2023 15:23
@Author  : dorianhgn
@FileName: DDPG Performance Evaluation.py
@Software: PyCharm
"""
import pandas as pd
import numpy as np

import tensorflow as tf
from tf_agents.environments import tf_py_environment

import Environment as envProjetMix

import data_processing as dp

# 1. Charger la politique de l'agent
policy_dir = 'policy_directory'  # Mettez à jour avec votre répertoire de politique
saved_policy = tf.compat.v2.saved_model.load(policy_dir)

# 2. Exécuter un épisode de l'environnement
data_clean = pd.read_csv('data/data_clean3.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')
prodSol = pd.read_csv('data/prod/prodSol.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')
prodEol = pd.read_csv('data/prod/prodEol.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')
prodRiv = pd.read_csv('data/prod/prodRiv.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')

env = envProjetMix.MixEnv(data_clean=data_clean[data_clean.index.year == 2022],
                          prodEol=prodEol[prodEol.index.year == 2022], prodRiv=prodRiv[prodRiv.index.year == 2022],
                          prodSol=prodSol[prodSol.index.year == 2022], pv=122, onshore=80,
                          offshore=13, random=False)

tf_env = tf_py_environment.TFPyEnvironment(env)

time_step = tf_env.reset()  # Initialiser l'environnement
policy_state = saved_policy.get_initial_state(tf_env.batch_size)

prev_obs_numpy = time_step.observation.numpy()
stocks = prev_obs_numpy[0, :4]
diff_stocks = np.zeros((1, 4), float)
biogas = np.array([0])  # np.array([prev_obs_numpy[0, -1]])
i = 0
max_index = env.max_index

while not time_step.is_last():
    i += 1
    policy_step = saved_policy.action(time_step, policy_state)
    policy_state = policy_step.state
    time_step = tf_env.step(policy_step.action)  # Appliquer l'action à l'environnement

    obs_numpy = time_step.observation.numpy()

    # Calcul de ce qui est dans les stocks, stocké et le biogas à retrancher pour avoir les flux
    stocks = np.vstack((stocks, obs_numpy[0, :4]))
    biogas = np.append(biogas, obs_numpy[0, -1])
    diff_stocks = np.vstack((diff_stocks,
                             (obs_numpy - prev_obs_numpy)[0, :4]
                             ))

    # update du prev_obs_numpy
    prev_obs_numpy = obs_numpy
    if i % int(max_index / 10) == 0:
        print(str(int(i / (max_index/100)) + 1) + '%')
        """if i / max_index/10 == 5:
            break"""

# Calcul du df de production résiduelle
prodRes = dp.prodRes(data_clean[data_clean.index.year == 2022]['Consommation (MW)'],
                     prodEol[prodEol.index.year == 2022]['Eolien (TW)'],
                     prodSol[prodSol.index.year == 2022]['Solaire (TW)'],
                     prodRiv[prodRiv.index.year == 2022]['Hydraulique (MW)'])

# Calcul du flux
techno_Spec = np.array([
    # etain, etaout, Q, S, vol
    [.95, .9, 9.3, 9.3, 180],  # PHS
    [.9, .95, 20.08, 20.08, 74.14],  # Battery
    [.59, .45, 32.93, 7.66, 125000],  # Methanation
    [1., 1., 10, 10, 2000]  # Lakes
])

diff_stocks[:, 2] -= biogas

indices_stock = diff_stocks >= 0  # Trouver les indices pour lesquels on stocke
indices_destock = diff_stocks < 0  # Trouver les indices pour lesquels on déstocke

flux_entrant = np.where(indices_stock, diff_stocks * techno_Spec[:, -1] / techno_Spec[:, 0], 0)  # on stocke
flux_sortant = np.where(indices_destock, diff_stocks * techno_Spec[:, -1] / techno_Spec[:, 0], 0)  # on déstocke

indices_dech = prodRes[:max_index] <= 0
pasTemp = 1 / 4  # 1/4 d'heure

MWh_non_fournis = - np.sum(
    prodRes[:i + 1][indices_dech] - np.sum(flux_entrant[1:, ][indices_dech], axis=1)
    - np.sum(flux_sortant[1:, ][indices_dech], axis=1)
) * pasTemp

print("Avec la police de l'agent, les GWh non fournies sur l'année 2022 avec les données de RTE sont de " +
      str(MWh_non_fournis / 1000) + ' GWh')
