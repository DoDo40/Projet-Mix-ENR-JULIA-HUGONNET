# Projet Mix Energétique
Auteurs : Manon JULIA & Dorian HUGONNET

## Requirements :
Python [3.8](https://www.python.org/downloads/release/python-380/) avec [Conda](https://www.anaconda.com/download/) installé

### Configuration de l'environnement
1. Ouvrez votre terminal ou invite de commande, selon votre système d'exploitation.
2. Assurez-vous d'avoir Conda installé et configuré correctement sur votre système.
3. Utilisez la commande suivante pour créer un nouvel environnement :
```shell  
conda create --name Projet_Mix_agents python=3.8
```  
4. Une fois l'installation terminée, vous pouvez activer l'environnement en utilisant la commande suivante :
-  Sous Windows :
```shell
conda activate Projet_Mix_agents
```

- Sous Linux :
```shell
source activate Projet_Mix_agents
```

5. Naviguez avec `cd` vers le dossier `"Projet Mix"`

6. Executez la commande suivante pour installer les packages :
```shell
pip install -r requirements.txt
```

## Correspondance entre les fichiers
#### DDPG Agent :
Le DDPG Agent utilise les données contenues dans le répertoire `/data`

Son environnement est le fichier `Environment.py`

La politique générée par cet agent est contenue dans le répertoire `policy_directory`

###### Entraînement de l'agent DDPG : 
Pour entrainer l'agent DDPG, il faut run `DDPG Training.py` puis run `DDPG Resume Training auto checkpoint.py`

Une erreur se produit à la fin d'une itération (qui tourne pendant 3-4h) du `DDPG Resume Training auto checkpoint.py`. 
Il faut donc le relancer à toutes les 3-4 heures (je n'ai pas eu le temps de corriger cette erreur).

Il serait idéal de l'entraîner avec les valeurs de `pv`, `onshore` et `offshore` calculées avec l'optimiseur de EOLES 
(cf le travail de Philippe Quirion).

###### Tester la politique de l'agent DDPG : 
Run le code `DDPG Performance Evaluation.py`. À titre comparatif, une politique d'action random donne un nombre de GWh
non fournis calculés en lançant le code : `Classic performance Evaluation.py`

Idéalement avec, les paramètres d'EOLES, le DDPG devrait s'approcher de zero s'il optimise au mieux ses actions. Mais
cela peut demander un très grand nombre d'entraînement, et un ajustement des paramètres important. De plus, utiliser un
autre agent (SAC, etc.) permettrai d'améliorer la politique.

#### DQN Agent : 
Le DQN Agent utilise les données contenues dans le répertoire `/data/Dossal/`

Son environnement d'entraînement est le fichier `Environment2050.py`

Son environnement de test est le fichier `Environment2050_test.py`

La politique générée par cet agent est contenue dans le répertoire `DQN_policy_directory`

D'autres politiques sont contenues dans Working_policies. Pour les utiliser il faut copier les données contenues dans 
`Working_policies/DQN_policy_directoryi` dans `DQN_policy_directory` afin de pouvoir run les tests de politique.


###### Entraînement et test de la politique de l'agent DQN : 

Pour entraîner l'agent, il faut run `DQN Training Checkpoint.py`

Soyez sûr que cette ligne est définie comme ceci :

```python
env = envProjetMix.MixEnv(Scenario=Scenario, prodRiv=prodRiv, prodSol=prodSol, prodOnshore=prodOnshore,
                          prodOffshore=prodOffshore, pv=122, onshore=80,
                          offshore=13, scenario="ADEME")
```

Vous pouvez ajuster ces hyperparamètres : 

```python
num_iterations = 13200  # Nombre total d'itérations à effectuer
initial_collect_steps = 1000  # Nombre d'itérations initiales pendant lesquelles l'agent joue de manière aléatoire
collect_steps_per_iteration = 1  # Nombre de pas de collecte par itération
replay_buffer_max_length = 200000  # Capacité maximale du replay buffer
batch_size = 2**15  # 2**9 # Taille du batch pour le training
learning_rate = 5e-5  # Taux d'apprentissage
log_interval = 200  # Fréquence des logs
num_eval_episodes = 10  # Nombre d'épisodes à évaluer
eval_interval = 1000  # Fréquence d'évaluation
```

Les paramètres qui dépendent de la puissance et de la mémoire vive de votre machine sont `replay_buffer_max_length` par 
rapport à la mémoire vive et `batch_size` par rapport à la puissance de calcul.

La chose à vérifier est la fonction `loss` qui correspond à l'écart de la valeur attendu par l'agent et celle délivrée
par l'environnement. Lorsqu'elle est grande, cela signifie que l'agent apprend. Mais lorsqu'elle explose, c'est mauvais 
signe. Généralement, la performance se dégrade.

Il faut donc reprendre tout l'entraînement en ajustant les hyperparamètres.

Pour vérifier vos résultats, vous pouvez à tout moment lancer `DQN Performance Evaluation.py` qui renvoie le nombre de 
GWh non fournies sur l'année 2050. À titre comparatif, la version brute de Mr. Dossal et Mme. Rondepierre peut être
observée en lançant le script `Classic DQN Perf Evaluation.py`.

Le résultat de notre statrégie vu en partie 1, est donnée en lançant le script `Strat DoDoMJ DQN Perf Evaluation.py`.

Tous les résultats de GWh non fournis sont comparables ici.