# 🧑‍💻 Adaptation de Domaine Non Supervisée pour la Ré-identification de Personnes avec Raffinement par A-Priori Caméra

Ce projet met en œuvre une pipeline d’**Adaptation de Domaine Non Supervisée (UDA)** pour la tâche de ré-identification de personnes (**Person Re-ID**).  
L’objectif est d’adapter un modèle pré-entraîné sur un jeu de données source (ex. *Market-1501*) à un jeu de données cible (*DukeMTMC-reID*) **sans utiliser les étiquettes de ce dernier**.

La contribution principale est une **nouvelle méthode de raffinement des pseudo-labels** qui exploite les informations des caméras afin de corriger les erreurs de clustering, améliorant ainsi la qualité de l’auto-apprentissage et les performances finales du modèle.

---

## ✨ Fonctionnalités Clés
- **Adaptation de Domaine Non Supervisée** : entraînement sur des données cibles non étiquetées.  
- **Pseudo-Étiquetage Progressif** : génération de pseudo-labels via clustering (*DBSCAN*).  
- **Raffinement par A-Priori Caméra** : pondération des votes des voisins selon la caméra pour corriger les erreurs.  
- **Stratégies d’Entraînement Avancées** :  
  - Perte *Triplet* avec minage des cas difficiles (*Hard Mining*).  
  - *Contrastive Mixup* pour enrichir l’espace des caractéristiques.  
  - *Random Identity Sampler* pour équilibrer les batches.  
- **Modèle Performant** : architecture *Vision Transformer (ViT)* via la bibliothèque [timm](https://github.com/huggingface/pytorch-image-models).

---

## 🚀 Résultats et Performances

| Modèle                                   | mAP (%) | Rank-1 (%) |
|------------------------------------------|---------|------------|
| Baseline (Pré-entraîné, sans adaptation) | 3.26    | 7.85       |
| Adapté (notre méthode avec raffinement)  | 43.81   | 64.09      |
| **Amélioration**                         | **+1242.8%** | **+716.0%** |

Ces résultats démontrent l’efficacité exceptionnelle du raffinement par caméra, transformant un modèle quasi-inutilisable en un système Re-ID performant.

---

## ⚙️ Pipeline d’Adaptation

Le processus se déroule en **3 étapes répétées à chaque époque** :

### 1️⃣ Génération de Pseudo-Labels
- Extraction des features de toutes les images du dataset cible.  
- Clustering avec *DBSCAN* pour regrouper les images d’une même personne.  
- Attribution d’un pseudo-label à chaque cluster.  
- Filtrage basé sur la confiance pour ne garder que les clusters fiables.  

### 2️⃣ Raffinement par A-Priori Caméra (**Innovation**)
- Hypothèse : une personne est plus susceptible d’être vue par plusieurs caméras.  
- Pour chaque image : analyse des `k` plus proches voisins.  
- Vote pondéré : un voisin d’une **caméra différente** compte plus qu’un voisin de la même caméra (`cross_cam_weight > 1.0`).  
- Permet de fusionner correctement des clusters séparés à tort.  

### 3️⃣ Entraînement Supervisé
- Utilisation des pseudo-labels raffinés comme vraies étiquettes.  
- Entraînement avec une fonction de perte *Triplet Loss*.  
- Itérations successives → pseudo-labels de plus en plus fiables → modèle plus performant.  

---

## 🛠️ Installation et Prérequis

### 1. Cloner le dépôt
```bash
git clone https://github.com/VOTRE_NOM/VOTRE_PROJET.git
cd VOTRE_PROJET
