# -*- coding: utf-8 -*-
"""
@Time    : 30/05/2023 17:31
@Author  : dorianhgn
@FileName: testing2050.py
@Software: PyCharm
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import pandas as pd

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import Environment2050 as envProjetMix

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

# Définition de l'environne
env = envProjetMix.MixEnv(Scenario=Scenario, prodRiv=prodRiv, prodSol=prodSol, prodOnshore=prodOnshore, prodOffshore=prodOffshore, pv=122, onshore=80,
                          offshore=13,scenario="ADEME")

print('action_spec:', env.action_spec())
print('time_step_spec.observation:', env.time_step_spec().observation)
print('time_step_spec.step_type:', env.time_step_spec().step_type)
print('time_step_spec.discount:', env.time_step_spec().discount)
print('time_step_spec.reward:', env.time_step_spec().reward)

# Validation de l'env avec tf_agents.environments.utils
utils.validate_py_environment(env, episodes=1)

# Wrapping up into a tf env
tf_env = tf_py_environment.TFPyEnvironment(env)

print(isinstance(tf_env, tf_environment.TFEnvironment))
print("TimeStep Specs:", tf_env.time_step_spec())
print("Action Specs:", tf_env.action_spec())
env.close()
