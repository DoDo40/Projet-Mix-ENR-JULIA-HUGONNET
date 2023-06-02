# -*- coding: utf-8 -*-
"""
@Time    : 16/05/2023 14:44
@Author  : dorianhgn
@FileName: data_processing.py
@Software: PyCharm
"""
# Import des librairies :

import pandas as pd
import numpy as np


# Definition de la classe Techno
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


# Definition de la fonction qui calcule la production résiduelle :
def prodRes(Conso, prodEol, prodSol, prodRiv, onshore=80, offshore=13,
            pv=122):  # Données DataFrame avec DateTime indexé (Datetime converti) avec les indices
    aux = prodSol * pv * 1000 + prodEol * (onshore + offshore) * 1000 + prodRiv - Conso
    return aux

