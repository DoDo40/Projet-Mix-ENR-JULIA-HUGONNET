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


# Définition de l'environnement du Mix énergétique
class MixEnv(py_environment.PyEnvironment):
    def __init__(self, data_clean, prodEol, prodRiv, prodSol, onshore, offshore, pv, pasTemp=0.25, random=True):
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
            shape=(4,), dtype=np.float32,  # 4 actions entre -1 et 1
            # minimum=-1., maximum=1.,  # entre -1 et 0 : décharge, entre 0 et 1 : charge
            minimum=[-1., -1., -1., -1.],
            maximum=[1., 1., 1., 0],
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
            [np.random.rand(), np.random.rand(), np.random.rand() * .2 + biogas, self.lake_inflows[0] / self.techno_Spec[3, -1]],
            self.data_scaled[self.current_index:self.current_index + 96, 10],
            self.data_scaled[self.current_index, :],
            [biogas]
        ))
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _calcul_flux(self, action):  # Calcul du flux utile (en électricité, même ordre de grandeur que prodRes)
        # entrant ou sortant à partir des actions que m'a données l'agent
        flux = np.zeros(len(self.techno_Spec))

        # Checker si les actions sont valides
        if action[3] > 0:
            raise ValueError('`action[3]` should be between -1 and 0')
        if np.any(action > 1) or np.any(action < -1):
            raise ValueError('`action` should be between -1 and 1')

        indices_stock = action > 0  # Trouver les indices pour lesquels on stocke
        indices_destock = action < 0  # Trouver les indices pour lesquels on déstocke

        # Calculer les flux pour l'action "stock"
        flux[indices_stock] = action[indices_stock] * self.techno_Spec[indices_stock, 3]

        # Calculer les flux pour l'action "destock"
        flux[indices_destock] = action[indices_destock] * self.techno_Spec[indices_destock, 2]

        return flux

    def _is_valid(self, flux):
        """
        Vérifie les flux individuellement, et les mets à 0 (pas d'action) ils ne sont pas valides
        :param flux: flux calculés à partir des actions de l'agent avec `_calcul_flux`
        :return: flux corrigé, pénalité
        """
        flux_valide = np.copy(flux)

        indices_negatifs = (flux < 0) & \
                           (- flux * self.pasTemp / self.techno_Spec[:, 1] > self._state[:4] * self.techno_Spec[:, -1])
        flux_valide = np.where(indices_negatifs, 0., flux)

        indices_positifs = (flux > 0) & (flux * self.pasTemp * self.techno_Spec[:, 0] >
                                         (1 - self._state[:4]) * self.techno_Spec[:, -1])
        flux_valide = np.where(indices_positifs, 0., flux_valide)

        neg_reward = np.sum(indices_positifs) + np.sum(indices_negatifs)  # Negative reward

        return flux_valide, neg_reward

    """def _calcul_flux_utile(self, flux):  # fonction qui transforme les actions de l'agent en ce qui est produit
        # depuis chaque technologie
        flux_utile = np.zeros(len(flux))

        indices_stock = flux <= 0  # Trouver les indices pour lesquels on stocke
        indices_destock = flux > 0  # Trouver les indices pour lesquels on déstocke

        flux_utile[indices_stock] = flux[indices_stock] * self.techno_Spec[:, 1]
        flux_utile[indices_destock] = flux[indices_destock] / self.techno_Spec[:, 0]

        return flux_utile"""  # Depreciated : self._calcul_flux calcule déjà le flux utile que l'on compare à prodRes

    def _update_state(self, flux_utile):
        """
        Fonction qui actualise les stocks
        :param flux_utile: ndarray avec les flux en électricité que l'on compare à prodRes
        :return: rien, on modifie directement `self._state`
        """
        # flux_entrant_sortant = np.zeros(len(flux_utile))

        indices_stock = flux_utile <= 0  # Trouver les indices pour lesquels on stocke
        indices_destock = flux_utile > 0  # Trouver les indices pour lesquels on déstocke

        flux_entrant = np.where(indices_stock, flux_utile / self.techno_Spec[:, 1], np.nan)
        flux_entrant_sortant = np.where(indices_destock, flux_utile * self.techno_Spec[:, 0], flux_entrant)
        self._state[:4] += flux_entrant_sortant / self.techno_Spec[:, -1] * self.pasTemp

        # Rajouter l'actualisation du methane selon une loi normale centrée sur la moyenne du biogas entrant
        biogas = np.random.normal(9.37, 5.0) * self.pasTemp / self.techno_Spec[2, -1]
        if self._state[2]+biogas < 1:
            self._state[2] += biogas
        else:
            biogas = 1. - self._state[2]
            self._state[2] = 1.

        # Rajouter les lacs
        if self.data['Jour'][self.current_index] == 1 and self.data['Heure'][self.current_index] == 0 \
                and self.data['Minute'][self.current_index] == 0:
            self._state[3] += self.lake_inflows[
                                 self.data['Mois'][self.current_index] - 1
                             ] / self.techno_Spec[3, -1]

        # Changer l'état de l'env
        self._state[4:] = np.concatenate((
            self.data_scaled[self.current_index:self.current_index + 96, 10],
            self.data_scaled[self.current_index, :],
            [biogas]
        ))
        pass

    def _neg_reward_stock(self, reward):
        """
        Si les stocks sont trop bas, alors il se prend une pénalité.
        Ceci est mis en place pour que l'agent ne déstocke trop
        :param reward:
        :return:
        """
        def func1(x):
            """
            Fonction qui calcule le %age de pénalité pour des stocks situés entre 0 et 20% pour les Batt et PHS
            :param x:
            :return:
            """
            return np.maximum(np.minimum(-1/.2 * x + 1, 1), 0)

        def func2(x):
            """
            Fonction qui calcule le %age de pénalité pour des stocks situés entre 0 et 5% pour le méthane
            :param x:
            :return:
            """
            return np.maximum(np.minimum(-1/.05 * x + 1, 1), 0)

        neg_reward_tab = func1(self._state[:2])
        neg_reward_tab = np.append(neg_reward_tab, func2(self._state[3]))
        reward -= np.sum(neg_reward_tab) * 33  # Facteur de pénalité à ajuster
        pass

    def _coef_prodRes(self, current_prodRes):
        """
        Fonction qui retourne un coefficient de récompense en fonction de prodRes
        Plus prodRes est négatif (donc plus il faut fournir), plus ce coef sera grand
        :param current_prodRes:
        :return:
        """
        return np.log(1-current_prodRes)

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

        # Calcul des flux entrant ou sortant à partir des actions que m'a données l'agent
        flux_preprocessed = self._calcul_flux(action)

        # 1. Verif des actions individuelles :
        flux, neg_reward = self._is_valid(flux_preprocessed)  # renvoie le flux modifié pour que ce soit valide et
        # les pénalités
        reward -= neg_reward * 10  # facteur de pénalité à ajuster

        # 3. Si PR < 0 (déstockage)
        if prodRes <= 0:
            somme_flux = np.sum(flux)
            neg_reward = prodRes - somme_flux
            if neg_reward > 0:  # il déstocke plus que nécessaire
                reward -= neg_reward * 1  # facteur de pénalité à ajuster
                self._update_state(flux)  # on actualise les stocks

            elif somme_flux > 0:  # il essaye de stocker alors que prodRes est négatif
                reward -= somme_flux * 10  # facteur de pénalité à ajuster
                # on n'actualise pas les stocks sinon pb
            else:
                self._update_state(flux)  # on actualise les stocks
                reward += somme_flux / prodRes * self._coef_prodRes(prodRes) * 35  # Récompense en fct de la
                # satisfaction de la demande, facteur de récompense à ajuster

        # 3. Si 0 < PR < flux max entrant (stockage ajusté)
        elif 0 < prodRes < self.flux_max_entrant:
            somme_flux = np.sum(flux)
            neg_reward = prodRes - somme_flux
            if neg_reward < 0:  # il stocke plus que ce qu'il y a dans PR
                # pénalité en fonction de ce qu'il stocke en trop
                reward += neg_reward * 5  # facteur de pénalité à ajuster
                # on normalise les flux en fonction de prodRes
                flux_norm = flux * prodRes / somme_flux
                self._update_state(flux_norm)  # on actualise les stocks avec les flux normalisés par rapport à prodRes
            else:
                self._update_state(flux)
                if somme_flux < 0:
                    reward += somme_flux * 1  # facteur de pénalité à ajuster

        else:  # Si PR > flux max entrant
            self._update_state(flux)
            somme_flux = np.sum(flux)
            if somme_flux < 0:
                reward += somme_flux * 1  # facteur de pénalité à ajuster
            # il fait ce qu'il veut, mais il vaut mieux qu'il stocke à fond sans dépasser les limites

        # Pénalité en fct de ce qu'il ne fournit pas(à mettre en place de manière à ce qu'il maximise sur le long terme)
        # Pour l'instant, ne pas la mettre car en soi, il aura moins de récompense s'il ne fournit pas assez

        # Negative reward en fct de l'état des stocks
        self._neg_reward_stock(reward)

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
