# -*- coding: utf-8 -*-
"""
@Time    : 15/05/2023 11:43
@Author  : dorianhgn
@FileName: Environment.py
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
import sklearn.preprocessing
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


# Definition des fonctions de charge et décharge d'une technologie
def load(tec, k, astocker):
    if astocker == 0:
        out = 0
    else:
        temp = min(astocker * tec.etain, tec.vol - tec.stored[k - 1], tec.S * tec.etain)
        tec.stored[k:] = tec.stored[k - 1] + temp
        out = astocker - temp / tec.etain
    return out


def unload(tec, k, aproduire):
    if aproduire == 0:
        out = 0
    else:
        temp = min(aproduire / tec.etaout, tec.stored[k], tec.Q / tec.etaout)
        if tec.name == 'Lake':
            tec.stored[k:int(endmonthlake[k])] = tec.stored[k] - temp
        else:
            tec.stored[k:] = tec.stored[k] - temp
        tec.prod[k] = temp * tec.etaout
        out = aproduire - tec.prod[k]
    return out


# Définition de l'environnement du Mix énergétique
class MixEnv(py_environment.PyEnvironment):
    def __init__(self, data_clean, prodEol, prodRiv, prodSol, onshore, offshore, pv, pasTemp=0.25, random=False):
        """
        Assurez vous que les données soient importés de cette façon :
        `data_clean = pd.read_csv('../data/data_clean.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')
        prodSol = pd.read_csv('../data/prod/prodSol.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')
        prodEol = pd.read_csv('../data/prod/prodEol.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')
        prodRiv = pd.read_csv('../data/prod/prodRiv.csv', sep=';', parse_dates=['Datetime']).set_index('Datetime')`
        :param data_clean: DataFrame
        :param prodEol: DataFrame
        :param prodRiv: DataFrame
        :param prodSol: DataFrame
        :param onshore: int (=80)
        :param offshore: int (=13)
        :param pv: int (=122)
        :param pasTemp:
        :param random:
        """

        # Appel au constructeur de la classe parente
        super(MixEnv, self).__init__()

        # Processing des données importées
        self.data = pd.merge(data_clean,
                             (prodSol * pv).merge((prodEol * (offshore + onshore)).merge(prodRiv, on='Datetime'),
                                                  on='Datetime'), on='Datetime')
        self.data['Production Residuelle (MW)'] = dp.prodRes(Conso=self.data['Consommation (MW)'],
                                                             prodEol=self.data['Eolien (TW)'],
                                                             prodSol=self.data['Solaire (TW)'],
                                                             prodRiv=self.data['Hydraulique (MW)'],
                                                             pv=1, onshore=1, offshore=1)  # prod data already processed
        self.data['Annee'] = self.data.index.year
        self.data['Mois'] = self.data.index.month
        self.data['Jour'] = self.data.index.day
        self.data['Heure'] = self.data.index.hour
        self.data['Minute'] = self.data.index.minute

        scaler = custom_scaler.CustomScaler()

        # Normalisation pour toutes les colonnes sauf 'Annee'
        data_normal = scaler.transform(self.data.drop(columns=['Annee']))

        # Normalisation personnalisée pour la colonne 'Annee'
        annee_min = 2000
        annee_max = 2100
        annee_normal = (self.data['Annee'] - annee_min) / (annee_max - annee_min) * 2 - 1

        # Concatenation des donnes normalisées et la colonne 'Annee' normalisée à la main avec `np.hstack`
        self.data_scaled = np.hstack((data_normal, annee_normal.to_numpy().reshape(-1, 1)))

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

        self.pasTemp = pasTemp  # 1/4 d'heure par défaut

        self.random = random  # Randomisation du début de l'entraînement
        if self.random:  # Si random = True, on indexe 'self.current_index' au hasard dans le reset
            self.starting_index = np.random.randint(0, len(self.data_scaled) - 96)
        else:
            self.starting_index = 0

        # lake_inflows provenant du csv de EOLES
        self.lake_inflows = np.array([1.3642965, 1.917242, 1.8321275, 1.418871, 1.0358125, 1.5900905, 1.1641765,
                                      0.9302595, 1.053223, 0.9381, 0.861544, 1.7097345]) * 1000

        # Définition des variables :
        self.max_index = len(self.data_scaled) - 96  # 96 = nombre de prévisions auquel l'agent a accès

        # Définition de l'état d'observation de mon agent
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(124,),  # vecteur de 123 valeurs observées
            dtype=np.float32,  # type des valeurs observées
            minimum=-1., maximum=1.,  # plage des valeurs observées
            name='observation')

        # Définition de l'état des actions que peut réaliser l'action
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int,  # Un action : choix de la stat des technos
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
        biogas = np.random.normal(9.37, 5.0) * self.pasTemp / self.techno_Spec[2, -1]
        self._state = np.concatenate((
            [np.random.rand(), np.random.rand(), np.random.rand() * .2 + biogas,
             self.lake_inflows[self.data['Mois'][self.current_index] - 1] / self.techno_Spec[3, -1]],
            self.data_scaled[self.current_index:self.current_index + 96, 10],
            self.data_scaled[self.current_index, :],
            [biogas]
        ))
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _update_state(self):

        # Rajouter l'actualisation du methane selon une loi normale centrée sur la moyenne du biogas entrant
        biogas = np.random.normal(9.37, 5.0) * self.pasTemp / self.techno_Spec[2, -1]
        if self._state[2] + biogas < 1:
            self._state[2] += biogas
        else:
            biogas = 1. - self._state[2]
            self._state[2] = 1.

        # Rajouter les lacs
        if self.data['Jour'][self.current_index] == 1 and self.data['Heure'][self.current_index] == 0 \
                and self.data['Minute'][self.current_index] == 0:
            self._state[3] = self.lake_inflows[
                                  self.data['Mois'][self.current_index] - 1
                                  ] / self.techno_Spec[3, -1]

        # Changer l'état de l'env
        self._state[4:] = np.concatenate((
            self.data_scaled[self.current_index:self.current_index + 96, 10],
            self.data_scaled[self.current_index, :],
            [biogas]
        ))
        pass

    # Fonctions de charge et de décharge des technologies
    def _load(self, prodRes, action):
        Astocker = prodRes
        for i in self.action_tec[action][0]:
            if Astocker != 0:
                temp = min(Astocker * self.techno_Spec[i, 0],
                           (1 - self._state[i]) * self.techno_Spec[i, 4],
                           self.techno_Spec[i, 3] * self.techno_Spec[i, 0]
                           )
                Astocker -= temp / self.techno_Spec[i, 0]
                self._state[i] += temp / self.techno_Spec[i, 4]
        pass

    def _unload(self, prodRes, action):
        Aproduire = -prodRes
        produit = 0
        compteur = 0
        for i in self.action_tec[action][0]:
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

        return produit

    def _perform_action(self, action):
        """
        Fonction qui réalise l'action lorsque l'on est au milieu de l'épisode
        :param action: action de l'agent
        :return: self._step, reward
        """
        self.current_index += 1
        # Initialisation de la reward
        reward = 0.

        # Obtention de 'prodRes'
        prodRes = self.data['Production Residuelle (MW)'][self.current_index] / 1000  # en GW (1000 MW)

        # Si PR < 0, on déstocke
        if prodRes <= 0:
            somme_flux = self._unload(prodRes, action)
            self._update_state()  # on actualise les stocks
            reward += somme_flux / prodRes * 1  # Récompense en fct de la
            # satisfaction de la demande, facteur de récompense à ajuster

        else:  # Si PR > 0, on stocke
            self._load(prodRes, action)
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
