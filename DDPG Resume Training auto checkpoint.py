# -*- coding: utf-8 -*-
"""
@Time    : 22/05/2023 15:23
@Author  : dorianhgn
@FileName: DDPG Resume Training.py
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

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.ddpg import actor_network
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import policy_saver

import Environment as envProjetMix

data_clean = pd.read_csv('data/data_clean3.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')
prodSol = pd.read_csv('data/prod/prodSol.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')
prodEol = pd.read_csv('data/prod/prodEol.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')
prodRiv = pd.read_csv('data/prod/prodRiv.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')

env = envProjetMix.MixEnv(data_clean=data_clean[data_clean.index.year != 2022],
                          prodEol=prodEol[prodEol.index.year != 2022], prodRiv=prodRiv[prodRiv.index.year != 2022],
                          prodSol=prodSol[prodSol.index.year != 2022], pv=122, onshore=80,
                          offshore=13)

tf_env = tf_py_environment.TFPyEnvironment(env)

print(isinstance(tf_env, tf_environment.TFEnvironment))
print("TimeStep Specs:", tf_env.time_step_spec())
print("Action Specs:", tf_env.action_spec())

# Paramètres
actor_fc_layers = (400, 300)
critic_obs_fc_layers = (400,)
critic_action_fc_layers = None
critic_joint_fc_layers = (300,)
ou_stddev = 0.2
ou_damping = 0.15
target_update_tau = 0.05
target_update_period = 5
dqda_clipping = None
td_errors_loss_fn = tf.keras.losses.MeanSquaredError()
gamma = 0.99
reward_scale_factor = 1.0
gradient_clipping = None
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
debug_summaries = False
summarize_grads_and_vars = False
train_step_counter = tf.Variable(0, dtype=tf.int32)
collect_steps_per_iteration = 2

# Choix des optimizer
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)

# Acteur
act_net = actor_network.ActorNetwork(  # actor_distribution_network.ActorDistributionNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=actor_fc_layers,
)

# Critique
crit_net = critic_network.CriticNetwork(
    (tf_env.observation_spec(), tf_env.action_spec()),
    observation_fc_layer_params=critic_obs_fc_layers,
    action_fc_layer_params=critic_action_fc_layers,
    joint_fc_layer_params=critic_joint_fc_layers,
)

# global_step = tf.compat.v1.train.get_global_step()

# Agent
agent = ddpg_agent.DdpgAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    actor_network=act_net,
    critic_network=crit_net,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    ou_stddev=ou_stddev,
    ou_damping=ou_damping,
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    dqda_clipping=dqda_clipping,
    td_errors_loss_fn=td_errors_loss_fn,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping,
    debug_summaries=debug_summaries,
    summarize_grads_and_vars=summarize_grads_and_vars,
    train_step_counter=train_step_counter,
)
agent.initialize()

# Buffer de replay
replay_buffer_capacity = 100000  # Ajustez en fonction de votre mémoire disponible

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=replay_buffer_capacity,
)

# Charger le point de contrôle
checkpoint_dir = 'checkpoint_directory'
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter
)
train_checkpointer.initialize_or_restore()

# Métriques
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

# Driver
collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch] + train_metrics,
    num_steps=collect_steps_per_iteration,
)

# Définition du policySaver :
policy_dir = 'policy_directory'
tf_policy_saver = policy_saver.PolicySaver(agent.collect_policy)

# Boucle d'entrainement
num_iterations = 700000  # Ajustez en fonction de votre problème
checkpoint_interval = 300

for _ in range(num_iterations):
    # Collecte des données
    collect_driver.run()

    # Entrainement de l'agent
    experience = replay_buffer.gather_all()
    train_loss = agent.train(experience)
    replay_buffer.clear()

    # Enregistrement de la politique de l'agent et du checkpoint à intervalles réguliers
    if agent.train_step_counter.numpy() % checkpoint_interval == 0:
        print(f"Step = {agent.train_step_counter.numpy()}, Loss = {train_loss.loss.numpy()}")

        # Enregistrement de la police
        tf_policy_saver.save(policy_dir)

        # Enregistrement du point de contrôle
        train_checkpointer.save(agent.train_step_counter.numpy())
"""
# Obtenir la politique de l'agent
policy = agent.policy

time_step = tf_env.reset()  # Initialiser l'environnement
policy_state = policy.get_initial_state(tf_env.batch_size)

while not time_step.is_last():
    action_step = policy.action(time_step, policy_state)  # Obtenir l'action de la politique
    policy_state = action_step.state
    time_step = tf_env.step(action_step.action)  # Appliquer l'action à l'environnement"""
