# -*- coding: utf-8 -*-
"""
@Time    : 18/05/2023 10:36
@Author  : dorianhgn
@FileName: testing.py
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

import Environment2 as envProjetMix

data_clean = pd.read_csv('data/data_clean3.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')
prodSol = pd.read_csv('data/prod/prodSol.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')
prodEol = pd.read_csv('data/prod/prodEol.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')
prodRiv = pd.read_csv('data/prod/prodRiv.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')

env = envProjetMix.MixEnv(data_clean=data_clean, prodEol=prodEol, prodRiv=prodRiv, prodSol=prodSol, pv=122, onshore=80,
                          offshore=13)

print('action_spec:', env.action_spec())
print('time_step_spec.observation:', env.time_step_spec().observation)
print('time_step_spec.step_type:', env.time_step_spec().step_type)
print('time_step_spec.discount:', env.time_step_spec().discount)
print('time_step_spec.reward:', env.time_step_spec().reward)

"""
# Validation manuelle de l'environnement
# Nombre d'épisodes à tester
num_episodes = 3

# Durée maximale d'un épisode (pour éviter une boucle infinie)
max_steps = 398000

for episode in range(num_episodes):
    # Réinitialisez l'environnement pour le début de l'épisode
    timestep = env.reset()
    print(f"Episode: {episode + 1}")
    print(timestep)

    for step in range(max_steps):
        if timestep.is_last():
            break
        # Choisissez une action de manière aléatoire (vous pouvez remplacer cette partie par votre propre politique)
        action = np.concatenate((np.random.uniform(-1, 1, 3), np.random.uniform(-1, 0, 1)))

        # Appliquez l'action à l'environnement
        timestep = env.step(action)

        # Imprimez les détails du timestep
        # print(f"Step: {step + 1}, Action: {action}, Reward: {timestep.reward}, New State: {timestep.observation}")

        # Affichez le numéro du step tous les 100 000 steps
        if (step + 1) % 100000 == 0:
            print(f"Step: {step + 1}, Action: {action}, Reward: {timestep.reward}, New State: {timestep.observation}")

# N'oubliez pas de fermer l'environnement à la fin
env.close()
"""

# Validation de l'env avec tf_agents.environments.utils
utils.validate_py_environment(env, episodes=1)

# Wrapping up into a tf env
tf_env = tf_py_environment.TFPyEnvironment(env)

print(isinstance(tf_env, tf_environment.TFEnvironment))
print("TimeStep Specs:", tf_env.time_step_spec())
print("Action Specs:", tf_env.action_spec())
env.close()
