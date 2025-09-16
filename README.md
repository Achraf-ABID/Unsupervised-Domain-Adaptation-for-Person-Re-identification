
# Adaptation de Domaine Non Supervisée pour la Ré-identification de Personnes avec Raffinement par A-Priori Caméra

Ce projet met en œuvre une puissante pipeline d'**Adaptation de Domaine Non Supervisée (Unsupervised Domain Adaptation - UDA)** pour la tâche de ré-identification de personnes (Person Re-ID). L'objectif est d'adapter un modèle pré-entraîné sur un jeu de données source (par ex., Market-1501) à un jeu de données cible (DukeMTMC-reID) **sans utiliser les étiquettes de ce dernier**.

La contribution principale de ce travail est une **nouvelle méthode de raffinement des pseudo-labels qui exploite les informations des caméras** pour corriger les erreurs de clustering, améliorant ainsi significativement la qualité de l'auto-apprentissage et les performances finales du modèle.

## ✨ Fonctionnalités Clés

*   **Adaptation de Domaine Non Supervisée** : Entraînement sur des données cibles non étiquetées en utilisant des pseudo-labels générés par clustering.
*   **Raffinement par A-Priori Caméra** : Une méthode innovante pour corriger les pseudo-labels en pondérant les votes des plus proches voisins en fonction de leur caméra d'origine.
*   **Pseudo-Étiquetage Progressif** : Le seuil de confiance pour accepter les pseudo-labels diminue progressivement, permettant au modèle de s'entraîner sur de plus en plus de données au fil du temps.
*   **Stratégies d'Entraînement Avancées** :
    *   Fonction de perte Triplet avec minage des cas difficiles (*Hard Mining*).
    *   Augmentation de données dans l'espace des caractéristiques avec *Contrastive Mixup*.
    *   Échantillonneur par identité (*Random Identity Sampler*) pour un entraînement par batchs plus efficace.
*   **Modèle Performant** : Basé sur une architecture Vision Transformer (ViT) chargée via la bibliothèque `timm`.

## 🚀 Résultats et Performances

L'approche développée montre une amélioration spectaculaire par rapport au modèle de base non adapté. En partant d'un modèle pré-entraîné sur Market-1501 et en l'adaptant à DukeMTMC-reID, nous obtenons les résultats suivants :

| Modèle                                         | mAP (%)             | Rank-1 (%)          |
| ---------------------------------------------- | ------------------- | ------------------- |
| **Baseline** (Pré-entraîné, sans adaptation)   | 3.26%               | 7.85%               |
| **Adapté** (Notre méthode avec raffinement)    | **43.81%**          | **64.09%**          |
| **Amélioration Relative**                      | **+1242.8%**        | **+716.0%**         |

Ces résultats démontrent l'efficacité exceptionnelle de la méthode d'adaptation, qui transforme un modèle initialement peu performant sur le domaine cible en un système de Re-ID robuste et précis.

## 🛠️ Guide d'Installation et d'Utilisation

### 1. Cloner le Dépôt

```bash
git clone https://github.com/VOTRE_NOM_UTILISATEUR/VOTRE_NOM_DE_PROJET.git
cd VOTRE_NOM_DE_PROJET
```

### 2. Créer l’environnement et installer les dépendances

Il est fortement recommandé d'utiliser un environnement virtuel (comme `venv` ou `conda`) pour isoler les dépendances du projet.

```bash
# Créez et activez votre environnement virtuel (exemple avec venv)
python -m venv venv
source venv/bin/activate  # Sur Windows, utilisez: venv\Scripts\activate

# Installez les paquets requis à partir du fichier requirements.txt
pip install -r requirements.txt
```

### 3. Fichier `requirements.txt`

Ce fichier est utilisé par la commande `pip install -r` et doit contenir les dépendances suivantes :

```txt
torch
torchvision
timm
scikit-learn
numpy
Pillow
tqdm
```

### 4. 📂 Organisation des Données

Pour que le script s'exécute sans erreur, vous devez organiser vos jeux de données et votre modèle pré-entraîné en respectant l'arborescence ci-dessous :

```
/chemin/vers/vos/donnees/
├── dukemtmcreid/
│   ├── bounding_box_train/
│   ├── bounding_box_test/
│   └── query/
│
└── market/
    └── Market1501_clipreid_12x12sie_ViT-B-16_60.pth
```

👉 **Important** : N'oubliez pas d'**adapter les chemins** (`DUKE_DATA_PATH`, `MARKET_MODEL_PATH`, etc.) dans la classe `Config` du script `train.py` pour qu'ils pointent vers les bons emplacements sur votre machine.

### ▶️ Lancement de l’Entraînement

Une fois la configuration terminée, lancez le processus d'adaptation avec la commande suivante :

```bash
python train.py
```

Le script exécutera automatiquement les étapes suivantes :
1.  **Évaluation du modèle de base** pour établir une performance de référence.
2.  **Lancement du processus d'adaptation itératif**, qui alterne entre la génération de pseudo-labels, leur raffinement et l'entraînement du modèle.
3.  **Sauvegarde du meilleur modèle** (ex. `best_model_camera_refined.pth`) à chaque fois que les performances de validation (mAP) s'améliorent.
4.  **Arrêt anticipé** (*Early Stopping*) si les performances n'augmentent plus.
5.  **Évaluation finale** du meilleur modèle sauvegardé à la fin du processus.

## 🔧 Paramètres Principaux

Tous les hyperparamètres clés peuvent être facilement modifiés directement dans la classe `Config` en haut du script `train.py`. Les plus importants incluent :

*   `ADAPTATION_EPOCHS` : Le nombre maximum d’époques pour le cycle d'adaptation.
*   `ADAPTATION_LR` : Le taux d’apprentissage pour l'optimiseur Adam.
*   `P` & `K` : Le nombre d’identités (`P`) et d’instances par identité (`K`) à inclure dans chaque batch.
*   `CONFIDENCE_THRESHOLD_START` / `_END` : Les seuils de confiance de départ et de fin pour le filtrage progressif des pseudo-labels.
*   `CAMERA_REFINEMENT_K` : Le nombre de plus proches voisins (`k`) à considérer lors de l'étape de raffinement par caméra.
*   `CAMERA_REFINEMENT_WEIGHT` : Le poids crucial appliqué aux votes des voisins provenant de caméras différentes (une valeur > 1.0 est recommandée pour valoriser la diversité des points de vue).

## 📜 Licence

Ce projet est distribué sous la licence MIT. Consultez le fichier `LICENSE` pour plus de détails.
```
