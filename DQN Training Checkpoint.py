# -*- coding: utf-8 -*-
"""
@Time    : 30/05/2023 19:30
@Author  : dorianhgn
@FileName: DQN Training.py
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

from tf_agents.environments import tf_py_environment

from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.utils import common

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

# Définition de l'environnement
env = envProjetMix.MixEnv(Scenario=Scenario, prodRiv=prodRiv, prodSol=prodSol, prodOnshore=prodOnshore,
                          prodOffshore=prodOffshore, pv=122, onshore=80,
                          offshore=13, scenario="ADEME")

tf_env = tf_py_environment.TFPyEnvironment(env)

num_iterations = 13200  # Nombre total d'itérations à effectuer
initial_collect_steps = 1000  # Nombre d'itérations initiales pendant lesquelles l'agent joue de manière aléatoire
collect_steps_per_iteration = 1  # Nombre de pas de collecte par itération
replay_buffer_max_length = 200000  # Capacité maximale du replay buffer
batch_size = 2**15  # 2**9 # Taille du batch pour le training
learning_rate = 5e-5  # Taux d'apprentissage
log_interval = 200  # Fréquence des logs
num_eval_episodes = 10  # Nombre d'épisodes à évaluer
eval_interval = 1000  # Fréquence d'évaluation

train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)

fc_layer_params = (100,)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)

for _ in range(initial_collect_steps):
    collect_driver.run()

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

agent.train = common.function(agent.train)

agent.train_step_counter.assign(0)

checkpoint_dir = 'path_to_your_checkpoint_directory'
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=5,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
)
train_checkpointer.initialize_or_restore()

# Récupération du compteur d'étapes d'entraînement
global_step = tf.compat.v1.train.get_global_step()

policy_dir = 'DQN_policy_directory'  # remplacer par le chemin réel
saver = policy_saver.PolicySaver(agent.policy)
saver.save(policy_dir)

for _ in range(num_iterations):

    for _ in range(collect_steps_per_iteration):
        collect_driver.run()

    experience, unused_info = next(iterator)
    # Compute loss and training step
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    # Log loss and other metrics
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    # Save checkpoint
    if step % eval_interval == 0:
        train_checkpointer.save(step)
        saver.save(policy_dir)

        # Evaluate policy
        # avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        # print('step = {0}: Average Return = {1}'.format(step, avg_return))

saver.save(policy_dir)
