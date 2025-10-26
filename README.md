

# Unsupervised Domain Adaptation for Person Re-Identification using Progressive Pseudo-Labeling and Camera Priors

This repository contains a PyTorch implementation of an advanced Unsupervised Domain Adaptation (UDA) pipeline for Person Re-Identification (Re-ID). The primary goal is to adapt a Re-ID model, pre-trained on a source dataset (e.g., Market-1501), to perform effectively on a new, unlabeled target dataset (e.g., DukeMTMC-reID) without requiring any manual annotation.

The script showcases a sophisticated methodology that combines iterative pseudo-labeling, confidence-based sample filtering, cluster refinement with camera-aware priors, and a powerful Vision Transformer (ViT) backbone to achieve significant performance improvements over the baseline model.

## 🚀 Key Features

- **Unsupervised Domain Adaptation**: Adapts a model to a new domain with zero labels, saving significant manual annotation effort.
- **Vision Transformer (ViT) Backbone**: Utilizes a powerful, pre-trained ViT model from the `timm` library for robust feature extraction.
- **Progressive Pseudo-Labeling**: Employs a DBSCAN clustering algorithm with a dynamic confidence threshold that adapts over epochs, starting with high-certainty samples and gradually including more complex ones.
- **Camera-Aware Label Refinement**: An innovative step that leverages camera IDs as a prior to refine pseudo-labels. It corrects clustering errors by giving higher weight to potential matches across different cameras, simulating real-world Re-ID scenarios.
- **Advanced Loss Function**: Incorporates a hard-mining triplet loss with an annealed temperature parameter, focusing the training on the most informative samples.
- **Cluster Merging**: Intelligently merges similar pseudo-identity clusters based on centroid similarity to consolidate fragmented identities.
- **Comprehensive Analysis**: Automatically generates and saves a suite of insightful plots to visualize performance evolution, training dynamics, and the impact of key algorithmic components.

## 📖 Methodology

The adaptation process is an iterative loop designed to progressively refine the model's understanding of the new target domain.

1.  **Baseline Evaluation**: The script begins by loading a model pre-trained on a source domain (Market-1501) and evaluates its initial, unadapted performance on the target domain (DukeMTMC-reID). This establishes a performance baseline.

2.  **Iterative Adaptation Loop**: For each adaptation epoch, the following steps are executed:
    a. **Feature Extraction**: The current model extracts features for all images in the target training set.
    b. **Progressive Pseudo-Labeling**:
        - Features are clustered using **DBSCAN**.
        - A **confidence score** (cosine similarity to the cluster centroid) is calculated for each sample.
        - Only samples exceeding a **dynamic confidence threshold** are assigned pseudo-labels. This threshold decreases over epochs, ensuring only high-quality labels are used initially.
    c. **Cluster Merging**: Clusters with high centroid similarity are merged to form more coherent identity groups.
    d. **Camera-Aware Refinement**: Pseudo-labels are further refined using a k-NN approach. For each labeled sample, its neighbors' labels are polled. Votes from neighbors captured by **different cameras** are given a higher weight, significantly improving label quality by correcting single-camera clustering mistakes.
    e. **Model Fine-Tuning**: A specialized `RandomIdentitySampler` creates batches containing multiple instances of the same pseudo-identity. The model is then trained on these batches using an **advanced triplet loss** to pull features of the same pseudo-identity closer and push others apart.

3.  **Evaluation & Model Selection**: After each epoch, the model is evaluated on the target domain's test set (query and gallery). The model achieving the highest mean Average Precision (mAP) is saved as the best model. Early stopping is used to prevent overfitting.

4.  **Final Analysis**: Once training is complete, the script provides a final comparison between the baseline and the best-adapted model and generates analytical graphs visualizing the entire process.

## 🏁 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch and Torchvision
- CUDA-enabled GPU (recommended for reasonable training times)

### 1. Clone the Repository

```bash
git clone https://github.com/achraf-abid/unsupervised-domain-adaptation-for-person-re-identification.git
cd unsupervised-domain-adaptation-for-person-re-identification
```

### 2. Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Prepare Data and Pre-trained Model

1.  **Target Dataset**: Download the **DukeMTMC-reID** dataset and place it in a `./data/` directory. The expected structure is:
    ```
    ./data/DukeMTMC-reID/
    ├── bounding_box_train/
    │   ├── 0001_c1_f0000001.jpg
    │   └── ...
    ├── query/
    │   ├── 0002_c2_f0000002.jpg
    │   └── ...
    └── bounding_box_test/
        ├── 0003_c3_f0000003.jpg
        └── ...
    ```

2.  **Pre-trained Model**: Download a Re-ID model pre-trained on the Market-1501 dataset. The script is configured for the `clipreid_12x12sie_ViT-B-16_60.pth` model. Create a `./pretrained_models/` directory and place the model file inside.

    *Note: If you use different paths, be sure to update them via the command-line arguments.*

## ⚡ How to Run

Execute the main script from the terminal. The default parameters are configured for effective performance but can be easily tuned.

```bash
python main.py
```

### Command-Line Arguments

You can customize the execution using the following arguments:

-   `--duke_data_path`: Path to the DukeMTMC-reID dataset (default: `./data/DukeMTMC-reID`).
-   `--market_model_path`: Path to the pre-trained Market-1501 model (default: `./pretrained_models/Market1501_clipreid_12x12sie_ViT-B-16_60.pth`).
-   `--output_path`: Directory to save the best model and output graphs (default: `./output`).
-   `--adaptation_epochs`: Maximum number of adaptation epochs (default: `40`).
-   `--adaptation_lr`: Learning rate for the Adam optimizer (default: `3.5e-5`).
-   `--p`: Number of distinct identities per training batch (default: `16`).
-   `--k`: Number of instances per identity in a batch (default: `4`)..
-   `--early_stopping_patience`: Number of epochs to wait for improvement before stopping (default: `8`).

## 📊 Results & Analysis

The script automatically generates a series of visualizations to provide a deep understanding of the adaptation process. Below are examples of the output graphs.

### 1. Performance Evolution Over Epochs
This graph tracks the Mean Average Precision (mAP) and Rank-1 accuracy on the target test set throughout the adaptation process. It clearly shows the model's learning progress and helps identify the best-performing epoch.

![Performance Evolution](https://github.com/Achraf-ABID/Unsupervised-Domain-Adaptation-for-Person-Re-identification/blob/main/output/graphs/evolution_performances.png?raw=true)

### 2. Comparison: Before vs. After Adaptation
This bar chart provides a direct comparison of the model's performance before any adaptation (baseline) and after the full unsupervised domain adaptation process. It highlights the significant improvements gained.

![Before vs After](https://github.com/achraf-abid/reid-project/blob/main/graphs/comparaison_avant_apres.png?raw=true)

### 3. Training Dynamics: Loss vs. Pseudo-Labels
This dual-axis plot visualizes the relationship between the average training loss and the number of confident pseudo-labels generated per epoch. It offers insights into the stability and effectiveness of the pseudo-labeling strategy.

![Training Dynamics](https://github.com/achraf-abid/reid-project/blob/main/graphs/dynamique_entrainement.png?raw=true)

### 4. Impact of Camera-Aware Refinement
This plot shows the percentage of pseudo-labels that were changed or corrected by the camera-aware refinement step at each epoch. It demonstrates the direct impact of this innovative technique on label quality.

![Refinement Impact](https://github.com/achraf-abid/reid-project/blob/main/graphs/impact_raffinement.png?raw=true)

---
