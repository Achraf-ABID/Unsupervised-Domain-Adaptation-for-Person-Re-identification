
# Unsupervised Domain Adaptation for Person Re-Identification with Camera-Aware Refinement

This project implements a powerful pipeline for **Unsupervised Domain Adaptation (UDA)** for the Person Re-Identification (Person Re-ID) task. The goal is to adapt a model pre-trained on a source dataset (e.g., Market-1501) to a target dataset (DukeMTMC-reID) **without using any of the target's labels**.

The main contribution of this work is a **novel pseudo-label refinement method that leverages camera information** to correct clustering errors, significantly improving the quality of self-training and the final performance of the model.

## ✨ Key Features

*   **Unsupervised Domain Adaptation**: Training on an unlabeled target dataset using clustering-based pseudo-labels.
*   **Camera-Aware Refinement**: An innovative method to correct pseudo-labels by weighting the votes of nearest neighbors based on their camera of origin.
*   **Progressive Pseudo-Labeling**: The confidence threshold for accepting pseudo-labels gradually decreases, allowing the model to train on more data over time.
*   **Advanced Training Strategies**:
    *   Triplet Loss with Hard Mining.
    *   Feature-space augmentation with Contrastive Mixup.
    *   Random Identity Sampler for more effective batch training.
*   **High-Performance Model**: Based on a Vision Transformer (ViT) architecture loaded via the `timm` library.

## 🚀 Results and Performance

The developed approach shows a dramatic improvement over the non-adapted baseline model. Starting with a model pre-trained on Market-1501 and adapting it to DukeMTMC-reID, we achieve the following results:

| Model                                    | mAP (%)             | Rank-1 (%)          |
| ---------------------------------------- | ------------------- | ------------------- |
| **Baseline** (Pre-trained, no adaptation) | 3.26%               | 7.85%               |
| **Adapted** (Our method with refinement) | **43.81%**          | **64.09%**          |
| **Relative Improvement**                 | **+1242.8%**        | **+716.0%**         |

These results demonstrate the exceptional effectiveness of the adaptation method, which transforms an initially poor-performing model on the target domain into a robust and accurate Re-ID system.

## 🛠️ Installation and Usage Guide

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_PROJECT_NAME.git
cd YOUR_PROJECT_NAME
```

### 2. Create Environment and Install Dependencies

It is highly recommended to use a virtual environment (like `venv` or `conda`) to isolate project dependencies.

```bash
# Create and activate your virtual environment (example with venv)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install the required packages from the requirements.txt file
pip install -r requirements.txt
```

### 3. `requirements.txt` File

This file is used by the `pip install -r` command and should contain the following dependencies:

```txt
torch
torchvision
timm
scikit-learn
numpy
Pillow
tqdm
```

### 4. 📂 Data Organization

For the script to run without errors, you must organize your datasets and pre-trained model according to the directory structure below:

```
/path/to/your/data/
├── dukemtmcreid/
│   ├── bounding_box_train/
│   ├── bounding_box_test/
│   └── query/
│
└── market/
    └── Market1501_clipreid_12x12sie_ViT-B-16_60.pth
```

👉 **Important**: Remember to **adapt the paths** (`DUKE_DATA_PATH`, `MARKET_MODEL_PATH`, etc.) in the `Config` class of the `train.py` script to point to the correct locations on your machine.

### ▶️ Running the Training

Once the setup is complete, launch the adaptation process with the following command:

```bash
python train.py
```

The script will automatically perform the following steps:
1.  **Evaluation of the baseline model** to establish a performance benchmark.
2.  **Launch of the iterative adaptation process**, which alternates between generating pseudo-labels, refining them, and training the model.
3.  **Saving the best model** (e.g., `best_model_camera_refined.pth`) whenever the validation performance (mAP) improves.
4.  **Early Stopping** if performance no longer increases.
5.  **Final evaluation** of the best saved model at the end of the process.

## 🔧 Key Parameters

All key hyperparameters can be easily modified directly in the `Config` class at the top of the `train.py` script. The most important ones include:

*   `ADAPTATION_EPOCHS`: The maximum number of epochs for the adaptation cycle.
*   `ADAPTATION_LR`: The learning rate for the Adam optimizer.
*   `P` & `K`: The number of identities (`P`) and instances per identity (`K`) to include in each batch.
*   `CONFIDENCE_THRESHOLD_START` / `_END`: The starting and ending confidence thresholds for the progressive filtering of pseudo-labels.
*   `CAMERA_REFINEMENT_K`: The number of nearest neighbors (`k`) to consider during the camera-aware refinement step.
*   `CAMERA_REFINEMENT_WEIGHT`: The crucial weight applied to votes from neighbors on different cameras (a value > 1.0 is recommended to prioritize diverse viewpoints).

## 📜 License

This project is distributed under the MIT License. See the `LICENSE` file for more details.
```
