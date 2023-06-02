# -*- coding: utf-8 -*-
"""
@Time    : 15/05/2023 11:43
@Author  : dorianhgn
@FileName: Environment2050_testing.py
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

# import de ma library contenant toutes les fonctions utiles pour la transformation des données
import data_processing as dp
from sklearn.preprocessing import MinMaxScaler
import custom_scaler


class Techno:
    def __init__(self, name, stored, prod, etain, etaout, Q, S, Vol):
        self.name = name  # nom de la techno
        self.stored = stored  # ce qui est stocké
        self.prod = prod  # ce qui est produit
        self.etain = etain  # coefficient de rendement à la charge
        self.etaout = etaout  # coefficient de rendement à la décharge
        self.Q = Q  # capacité installée (flux sortant)
        self.S = S  # capacité de charge d'une techno de stockage (flux entrant)
        self.vol = Vol  # Capacité maximale de stockage de la techno (Volume max)


# Définition de l'environnement du Mix énergétique
class MixEnv(py_environment.PyEnvironment):
    def __init__(self, Scenario, prodRiv, prodSol, prodOnshore, prodOffshore,
                 onshore, offshore, pv, scenario, pasTemp=1, random=False):
        """

        :param Scenario: Dictionnaire avec les scenarios
        :param prodRiv:
        :param prodSol:
        :param prodOnshore:
        :param prodOffshore:
        :param onshore:
        :param offshore:
        :param pv:
        :param scenario: scénario selectionné, par exemple : `"RTE"`
        :param pasTemp:
        :param random:
        """

        # Appel au constructeur de la classe parente
        super(MixEnv, self).__init__()

        # Processing des données importées
        self.data_clean = Scenario[scenario]

        self.data_clean = pd.concat(
            [self.data_clean, (prodSol * 122), (prodOnshore * 80), (prodOffshore * 13), prodRiv * 13], axis=1)

        self.data_clean['Jour'] = pd.to_datetime(self.data_clean.index, unit='h').day
        self.data_clean['Mois'] = pd.to_datetime(self.data_clean.index, unit='h').month
        self.data_clean['Heure'] = pd.to_datetime(self.data_clean.index, unit='h').hour

        data_clean_numpy = self.data_clean.to_numpy()
        self.data_clean['prodRes'] = - data_clean_numpy[:, 0] + data_clean_numpy[:, 1] + data_clean_numpy[:, 2] \
                                     + data_clean_numpy[:, 3] + data_clean_numpy[:, 4]

        # Normalisation des données
        scaler = MinMaxScaler()
        self.data_scaled = scaler.fit_transform(self.data_clean)

        if scenario == "RTE":
            constant_column = np.zeros((self.data_scaled.shape[0], 1))
        elif scenario == "ADEME":
            constant_column = np.ones((self.data_scaled.shape[0], 1)) * 0.5
        elif scenario == "Negawatt":
            constant_column = np.ones((self.data_scaled.shape[0], 1))
        else:
            raise ValueError("scenario n'est pas bon")
        self.data_scaled = np.concatenate((self.data_scaled, constant_column), axis=1)

        # Définition des constantes :
        self.techno_Spec = np.array([
            # etain, etaout, Q, S, vol (en GW)
            [.95, .9, 9.3, 9.3, 180],  # PHS
            [.9, .95, 20.08, 20.08, 74.14],  # Battery
            [.59, .45, 32.93, 7.66, 125000],  # Methanation
            [1., 1., 10, 10, 2000]  # Lakes
        ])

        self.action_tec = np.array([
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ])

        self.flux_max_entrant = 37.76

        self.pasTemp = pasTemp  # une heure par défaut

        self.produit = 0
        self.charge = 0

        self.random = random  # Randomisation du début de l'entraînement
        if self.random:  # Si random = True, on indexe 'self.current_index' au hasard dans le reset
            self.starting_index = np.random.randint(0, len(self.data_scaled))
        else:
            self.starting_index = 0

        # lake_inflows provenant du csv de EOLES
        self.lake_inflows = np.array([1.3642965, 1.917242, 1.8321275, 1.418871, 1.0358125, 1.5900905, 1.1641765,
                                      0.9302595, 1.053223, 0.9381, 0.861544, 1.7097345]) * 1000

        # Définition des variables :
        self.max_index = len(self.data_scaled) - 1

        # Définition de l'état d'observation de mon agent
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(15,),  # vecteur de 14 valeurs observées
            dtype=np.float32,  # type des valeurs observées
            minimum=0, maximum=1.,  # plage des valeurs observées
            name='observation')

        # Définition de l'état des actions que peut réaliser l'action
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int,  # Une action : choix de la stat des technos
            minimum=0,
            maximum=5,
            name='action')

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # Cf. data_processing.reset_state()
        self.current_index = self.starting_index
        self._episode_ended = False
        biogas = np.maximum(np.random.normal(9.37, 5.0), 0) * self.pasTemp / self.techno_Spec[2, -1]
        self._state = np.concatenate((
            [1., 1., 0.5 + biogas,
             self.lake_inflows[self.data_clean['Mois'][self.current_index] - 1] / self.techno_Spec[3, -1]],
            self.data_scaled[self.current_index, :],
            [biogas]
        ))
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _update_state(self):
        # Rajouter l'actualisation du méthane selon une loi normale centrée sur la moyenne du biogas entrant
        biogas = np.maximum(np.random.normal(9.37, 5.0), 0) * self.pasTemp / self.techno_Spec[2, -1]
        if self._state[2] + biogas < 1:
            self._state[2] += biogas
        else:
            biogas = 1. - self._state[2]
            self._state[2] = 1.

        # Rajouter les lacs
        if self.data_clean['Jour'][self.current_index] == 1 and self.data_clean['Heure'][self.current_index] == 0:
            self._state[3] = self.lake_inflows[
                                 self.data_clean['Mois'][self.current_index] - 1
                                 ] / self.techno_Spec[3, -1]

        # Changer l'état de l'env
        self._state[4:] = np.concatenate((
            self.data_scaled[self.current_index, :],
            [biogas]
        ))

        indices_zeros = self._state[:4] < 0
        indices_ones = self._state[:4] > 1

        self._state[:4] = np.where(indices_zeros, 0, self._state[:4])
        self._state[:4] = np.where(indices_ones, 1, self._state[:4])
        pass

    # Fonctions de charge et de décharge des technologies
    def _load(self, prodRes, action):
        Astocker = prodRes
        charge = 0
        for i in self.action_tec[action]:
            if Astocker != 0:
                temp = min(Astocker * self.techno_Spec[i, 0],
                           (1 - self._state[i]) * self.techno_Spec[i, 4],
                           self.techno_Spec[i, 3] * self.techno_Spec[i, 0]
                           )
                Astocker -= temp / self.techno_Spec[i, 0]
                self._state[i] += temp / self.techno_Spec[i, 4]
                charge += temp / self.techno_Spec[i, 0]
                if self._state[i] > 1:
                    self._state[i] = 1
        return charge

    def _unload(self, prodRes, action):
        Aproduire = -prodRes
        produit = 0
        compteur = 0
        for i in self.action_tec[action]:
            if Aproduire != 0:

                if compteur == 3:
                    temp = min(Aproduire / self.techno_Spec[3, 1],
                               self._state[3] * self.techno_Spec[3, 4],
                               self.techno_Spec[3, 2] / self.techno_Spec[3, 1]
                               )
                    self._state[3] -= temp / self.techno_Spec[3, 4]
                    produit += temp * self.techno_Spec[3, 1]
                    Aproduire -= temp * self.techno_Spec[3, 1]

                temp = min(Aproduire / self.techno_Spec[i, 1],
                           self._state[i] * self.techno_Spec[i, 4],
                           self.techno_Spec[i, 2] / self.techno_Spec[i, 1]
                           )
                self._state[i] -= temp / self.techno_Spec[i, 4]
                produit += temp * self.techno_Spec[i, 1]
                Aproduire -= temp * self.techno_Spec[i, 1]
                compteur += 1
                if self._state[i] < 0:
                    self._state[i] = 0

        return produit

    def _perform_action(self, action):
        """
        Fonction qui réalise l'action lorsque l'on est au milieu de l'épisode
        :param action: action de l'agent
        :return: self._step, reward
        """
        self.produit = 0
        self.charge = 0

        self.current_index += 1
        # Initialisation de la reward
        reward = 0.

        # Obtention de 'prodRes'
        prodRes = self.data_clean['prodRes'][self.current_index]  # en GW (1000 MW)

        # Si PR < 0, on déstocke
        if prodRes <= 0:
            self.produit = self._unload(prodRes, action)
            self._update_state()  # on actualise les stocks
            reward += self.produit / prodRes * 1  # Récompense en fct de la
            # satisfaction de la demande, facteur de récompense à ajuster

        else:  # Si PR > 0, on stocke
            self.charge = self._load(prodRes, action)
            self._update_state()

        return self._state, reward

    def _is_terminal_state(self):
        """
        check if we have reached the terminal state
        :return: True or False depending on 'self.current_index'
        """
        if self.random:
            if self.current_index == self.max_index:
                self.current_index = 0
                return False
            elif self.current_index == self.starting_index - 1:
                self.reset()
                return True
            else:
                return False
        elif not self.random and self.current_index == self.max_index:
            self.reset()
            return True
        else:
            return False

    def _step(self, action):
        """
        Fonction step de l'environnement
        :param action: action de l'agent
        :return:
        """
        if self._current_time_step is None or self._current_time_step.is_last():
            # If the current step is None, or it's the last step,
            # it means we need to start a new episode.
            return self.reset()

        # If we reach here, it means we're in the middle of an episode.

        # First, perform the action and update the state and reward.
        self._state, reward = self._perform_action(action)

        # Then, check if we have reached the terminal state.
        if self._is_terminal_state():
            print('terminado')
            self._episode_ended = True
            return ts.termination(np.array(self._state, dtype=np.float32), reward)

        # If we reach here, it means we're not at the end of the episode.
        return ts.transition(np.array(self._state, dtype=np.float32), reward, discount=1.0)
