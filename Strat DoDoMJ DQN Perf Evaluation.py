# -*- coding: utf-8 -*-
"""
@Time    : 30/05/2023 21:25
@Author  : dorianhgn
@FileName: Classic DQN Perf Evaluation.py
@Software: PyCharm
"""
# -*- coding: utf-8 -*-
"""
@Time    : 30/05/2023 20:29
@Author  : dorianhgn
@FileName: DQN Performance Evaluation.py
@Software: PyCharm
"""

import pandas as pd
import numpy as np

import tensorflow as tf
from tf_agents.environments import tf_py_environment

import Environment2050_testing as envProjetMix

# Import des demandes energétiques des scénarios de 2050
ADEME = pd.read_csv("data/Dossal/demand2050_ADEME.csv", header=None)
ADEME.columns = ["heures", "demande"]
RTE = pd.read_csv("data/Dossal/demand2050_RTE.csv", header=None)
RTE.columns = ["heures", "demande"]
Negawatt = pd.read_csv("data/Dossal/demand2050_negawatt.csv", header=None)
Negawatt.columns = ["heures", "demande"]
Scenario = {"RTE": RTE.demande, "ADEME": ADEME.demande, "Negawatt": Negawatt.demande}

# Import des productions de 2006
vre2006 = pd.read_csv("data/Dossal/vre_profiles2006.csv", header=None)
vre2006.columns = ["vre", "heure", "prod2"]

N = 8760
prod2006 = vre2006.prod2
prod2006_offshore = prod2006[0:N]
prod2006_onshore = prod2006[N:2 * N]
prod2006_pv = prod2006[2 * N:3 * N]

# Import des rivières
river = pd.read_csv("data/Dossal/run_of_river.csv", header=None)
river.columns = ["heures", "prod2"]
rivprod = river.prod2

# Ajustement de quelques paramètres
prodSol = prod2006_pv.reset_index(drop=True)
prodOnshore = prod2006_onshore.reset_index(drop=True)
prodOffshore = prod2006_offshore.reset_index(drop=True)
prodRiv = rivprod.reset_index(drop=True)

scenario = "ADEME"
# Définition de l'environnement
env = envProjetMix.MixEnv(Scenario=Scenario, prodRiv=prodRiv, prodSol=prodSol, prodOnshore=prodOnshore,
                          prodOffshore=prodOffshore, pv=122, onshore=80,
                          offshore=13, scenario=scenario)

tf_env = tf_py_environment.TFPyEnvironment(env)

time_step = tf_env.reset()  # Initialiser l'environnement

prev_obs_numpy = time_step.observation.numpy()
stocks = prev_obs_numpy[0, :4]
diff_stocks = np.zeros((1, 4), float)
biogas = np.array([0])  # np.array([prev_obs_numpy[0, -1]])
i = 0
max_index = env.max_index

prod_stock = np.array([])

# Calcul du df de production résiduelle
prodRes = env.data_clean['prodRes'].to_numpy()
prod_res_negatif = np.where(prodRes < 0, prodRes, 0)

while not time_step.is_last():
    i += 1
    if prodRes[i] > 0:
        if (0 <= prodRes[i] < 37.74):
            if prodRes[i] - prodRes[i-1] < 9.7:
                action = 0

            else:
                action = 4

        else:
            action = 4
    else:
        action = 0


    time_step = tf_env.step(action)  # Appliquer l'action à l'environnement

    obs_numpy = time_step.observation.numpy()

    prod_stock = np.append(prod_stock, env.produit)

    if i % int(max_index / 10) == 0:
        print(str(int(i / (max_index / 100)) + 1) + '%')
        """if i / max_index/10 == 5:
            break"""

# Calcul du flux
techno_Spec = np.array([
    # etain, etaout, Q, S, vol
    [.95, .9, 9.3, 9.3, 180],  # PHS
    [.9, .95, 20.08, 20.08, 74.14],  # Battery
    [.59, .45, 32.93, 7.66, 125000],  # Methanation
    [1., 1., 10, 10, 2000]  # Lakes
])

MWh_non_fournis = np.sum(-prod_stock - prod_res_negatif[:i])

print("Avec la strat Manon+DoDo, les GWh non fournies sur l'année 2022 avec les données de " + scenario + " sont de " +
      str(MWh_non_fournis) + ' GWh')
