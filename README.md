# Unsupervised Domain Adaptation for Person Re-identification with Vision Transformers

This repository contains the code for an advanced Unsupervised Domain Adaptation (UDA) approach for person re-identification (ReID). It leverages a Vision Transformer (ViT) model and a sophisticated pseudo-labeling strategy that is enhanced with camera-aware refinement to achieve significant performance improvements when adapting from a labeled source dataset to an unlabeled target dataset.
## 🎞️ Demo Preview

![Demo of UDA Process](assets/AI_Learns_to_See_in_a_New_City.gif)

## Project Overview

Person Re-identification (ReID) is a critical task in computer vision that involves identifying the same person across different cameras. While supervised methods perform well, they require extensive labeled data, which is often impractical to obtain for every new camera network. Unsupervised Domain Adaptation (UDA) addresses this by adapting a model trained on a labeled "source" dataset to an unlabeled "target" dataset.

This project implements a UDA pipeline that uses a Vision Transformer (ViT-B-16) backbone, pre-trained on a large-scale dataset, and fine-tunes it on the target domain using a progressive pseudo-labeling and clustering approach. A key innovation of this work is the **camera-aware label refinement** step, which leverages camera information to improve the quality of the generated pseudo-labels, leading to more robust and accurate models.

### Key Features:

*   **Unsupervised Domain Adaptation**: Adapts a ReID model from a labeled source to an unlabeled target dataset.
*   **Vision Transformer Backbone**: Utilizes the powerful ViT-B-16 architecture for feature extraction.
*   **Progressive Pseudo-Labeling**: Employs DBSCAN clustering to generate pseudo-labels for the target data, with a confidence threshold that adapts over epochs.
*   **Camera-Aware Refinement**: A novel technique that refines pseudo-labels by giving more weight to votes from neighbors captured by different cameras, which helps to correct noisy labels.
*   **Advanced Training Techniques**: Incorporates techniques like mixup augmentation, hard mining triplet loss, and an adaptive temperature to improve training stability and performance.

## Datasets

This project uses three standard large-scale person ReID datasets:

*   **Market-1501**: Contains 32,668 bounding boxes of 1,501 identities, captured by six cameras. It is a widely used benchmark for person ReID.
*   **DukeMTMC-reID**: A subset of the DukeMTMC dataset, it consists of 36,411 bounding boxes for 1,812 identities across eight cameras. It includes 16,522 training images of 702 identities, 2,228 query images, and 17,661 gallery images.
*   **MSMT17**: A more challenging dataset with 126,441 bounding boxes of 4,101 identities, captured by a 15-camera network in both indoor and outdoor environments, spanning a long period with complex lighting variations.

## Methodology

The UDA process in this notebook can be broken down into the following steps for each adaptation epoch:

1.  **Feature Extraction**: The current model is used to extract features for all images in the unlabeled target training set.
2.  **Progressive Pseudo-Labeling**:
    *   The DBSCAN clustering algorithm is applied to the extracted features to group similar instances.
    *   A confidence score is calculated for each sample within a cluster based on its similarity to the cluster's centroid.
    *   Pseudo-labels are assigned only to samples that meet a progressively adjusted confidence threshold. This threshold starts high and decreases over time, gradually incorporating more samples into the training process.
3.  **Camera-Aware Label Refinement**:
    *   For each labeled sample, its k-nearest neighbors are identified.
    *   A weighted voting process is used to determine if the sample's current pseudo-label should be changed. Votes from neighbors in different camera views are given a higher weight, based on the assumption that cross-camera matches are more informative for ReID.
4.  **Model Training**: The model is fine-tuned for one epoch using the refined pseudo-labels. The training process uses an advanced triplet loss with hard mining and an adaptive temperature, as well as mixup augmentation.
5.  **Evaluation and Model Saving**: The model's performance is evaluated on the target dataset's query and gallery sets using mean Average Precision (mAP) and Rank-1 accuracy. If the performance improves, the model is saved. Early stopping is used to conclude the training if there is no improvement for a set number of epochs.

## Experiments and Results

The notebook presents two main UDA experiments, a supervised baseline, and visualizations of the results.

### Experiment 1: Market-1501 to DukeMTMC-reID

*   **Source Dataset**: Market-1501 (for pre-training the baseline model)
*   **Target Dataset**: DukeMTMC-reID

| Model | mAP | Rank-1 Accuracy |
| :--- | :--- | :--- |
| **Baseline (No Adaptation)** | 3.26% | 7.85% |
| **Adapted Model** | **43.81%** | **64.09%** |

The adaptation process resulted in a remarkable **+1242.8%** improvement in mAP and a **+716.0%** increase in Rank-1 accuracy, demonstrating the effectiveness of the proposed UDA strategy.

### Experiment 2: MSMT17 to DukeMTMC-reID

*   **Source Dataset**: MSMT17 (for pre-training the baseline model)
*   **Target Dataset**: DukeMTMC-reID

| Model | mAP | Rank-1 Accuracy |
| :--- | :--- | :--- |
| **Baseline (No Adaptation)** | 3.26% | 7.85% |
| **Adapted Model** | **43.16%** | **62.97%** |

When adapting from the larger and more diverse MSMT17 dataset, the model achieved a **+1222.8%** improvement in mAP and a **+701.7%** increase in Rank-1 accuracy.

### Supervised Training on DukeMTMC-reID (for comparison)

A supervised training experiment was also conducted on the DukeMTMC-reID dataset to provide an upper-bound reference for the UDA performance.

| Model | mAP | Rank-1 Accuracy |
| :--- | :--- | :--- |
| **Supervised Model** | **70.43%** | **83.44%** |

### Visualizations

The notebook includes several plots to visualize the training dynamics and results:
*   A line chart showing the evolution of mAP and Rank-1 accuracy during the adaptation process.
*   A bar chart comparing the performance of the baseline model with the final adapted model.
*   A line chart illustrating the relationship between the training loss and the number of pseudo-labeled samples over epochs.
*   A bar chart showing the percentage of labels modified by the camera-refinement step in each epoch.

## How to Use

To run this notebook, you will need a Python environment with the following libraries installed:
`scikit-learn`, `timm`, `torch`, `torchvision`, `numpy`, `Pillow`, `tqdm`, `matplotlib`, and `seaborn`.

The notebook is structured to be executed sequentially. Simply run the cells in order to perform the experiments. The configuration parameters in the `Config` class can be adjusted to explore different settings for the UDA process.

## Conclusion

This project showcases a highly effective approach to Unsupervised Domain Adaptation for Person Re-identification. By combining a powerful Vision Transformer model with a progressive pseudo-labeling strategy and a novel camera-aware refinement technique, it achieves substantial performance gains on challenging cross-domain ReID tasks. The detailed experiments and visualizations provide valuable insights into the dynamics of the adaptation process.
