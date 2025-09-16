Unsupervised Domain Adaptation for Person Re-identification
![alt text](https://img.shields.io/github/languages/top/Achraf-ABID/Unsupervised-Domain-Adaptation-for-Person-Re-identification)
![alt text](https://img.shields.io/badge/license-MIT-blue.svg)
This project implements an Unsupervised Domain Adaptation (UDA) method for Person Re-identification, featuring a novel pseudo-label refinement strategy that leverages camera information to correct clustering errors. This approach dramatically improves the model's performance on a new, unlabeled target dataset by generating higher-quality training signals.
Table of Contents
Overview
Key Features
Methodology
Results
Installation
How to Run
Configuration
License
Overview
Person Re-identification (Re-ID) aims to match images of the same person across different camera views. While supervised methods achieve high accuracy, they require large-scale labeled datasets, which are expensive and time-consuming to create for every new environment.
Unsupervised Domain Adaptation (UDA) offers a solution by adapting a model trained on a labeled source domain (e.g., Market-1501) to an unlabeled target domain (e.g., DukeMTMC-reID). This project focuses on a UDA pipeline that uses iterative pseudo-labeling. Its core contribution is a Camera-Aware Refinement technique that uses camera IDs as a powerful heuristic to clean noisy pseudo-labels, leading to a more robust and accurate model.
Key Features
Unsupervised Domain Adaptation: No manual labeling is required for the target dataset.
Clustering-Based Pseudo-Labeling: Uses DBSCAN to generate initial identity labels for the target training set.
Novel Camera-Aware Refinement: Corrects and merges pseudo-labels by giving more weight to matches of an individual across different cameras. This is based on the assumption that inter-camera matches are stronger evidence of a correct ID.
Advanced Training Strategies:
Vision Transformer (ViT) backbone for powerful feature extraction.
Hard-Mining Triplet Loss to focus training on difficult examples.
Progressive Confidence Throttling to ensure only high-quality pseudo-labels are used in early training stages.
Methodology
The adaptation process is iterative. In each epoch, the model is refined through the following three stages:
Pseudo-Label Generation: The current model extracts features from all images in the target dataset. DBSCAN is then applied to these features to cluster them into pseudo-identity labels. A confidence threshold filters out noisy and uncertain assignments.
Camera-Aware Refinement: This is the key innovation. For each sample with a pseudo-label, we analyze its k-nearest neighbors. A weighted voting system determines if the label should be kept or changed:
A neighbor from a different camera gets a higher vote weight.
A neighbor from the same camera gets a lower vote weight.
This process corrects common clustering errors, such as assigning different IDs to the same person who appears in multiple camera views.
Model Training: The model is fine-tuned for one epoch using the newly refined pseudo-labels. A specialized sampler ensures that each batch contains multiple images of the same pseudo-identity, enabling effective metric learning with a triplet loss function.
This cycle repeats, allowing the model to generate progressively better features and, consequently, more accurate pseudo-labels.
Results
This method shows a significant improvement over the baseline model (pre-trained on the source domain without any adaptation).
Model	mAP	Rank-1
Baseline (No Adaptation)	~3-5%	~7-10%
Adapted Model (with Camera Refinement)	~43%	~64%
(Results are approximate based on typical performance on the DukeMTMC-reID dataset).		
Installation
Clone the repository:
code
Bash
git clone https://github.com/Achraf-ABID/Unsupervised-Domain-Adaptation-for-Person-Re-identification.git
cd Unsupervised-Domain-Adaptation-for-Person-Re-identification
Install the required Python packages. It is recommended to use a virtual environment.
code
Bash
pip install -r requirements.txt
Note: If a requirements.txt is not available, you can install the main libraries manually:
code
Bash
pip install torch torchvision timm scikit-learn numpy
How to Run
Download Datasets: Download the source (e.g., Market-1501) and target (e.g., DukeMTMC-reID) datasets.
Get Pre-trained Model: Download a Re-ID model pre-trained on the source dataset.
Configure Paths: Open the main Python script and update the paths in the Config class to point to your datasets, pre-trained model, and desired output directory.
code
Python
class Config:
    DRIVE_BASE_PATH = "/path/to/your/output_directory/"
    DUKE_DATA_PATH = "/path/to/your/dukemtmcreid_dataset/"
    MARKET_MODEL_PATH = "/path/to/your/pretrained_model.pth"
    # ...
Execute the Script: Run the main script to start the adaptation process.
code
Bash
python main.py
The script will first evaluate the baseline performance, then start the training loop. Progress will be printed to the console, and the best-performing model will be saved automatically.
Configuration
All major hyperparameters and settings can be adjusted in the Config class at the top of the script. This includes:
Learning Rate (ADAPTATION_LR)
Batch Size (P, K)
Number of Epochs (ADAPTATION_EPOCHS)
DBSCAN and Clustering Parameters
Camera Refinement Weight (CAMERA_REFINEMENT_WEIGHT)
License
This project is licensed under the MIT License. See the LICENSE file for details.
