Of course. Here is a professional `README.md` for your project, written in English.

---

# Domain Adaptation for Person Re-Identification

This project implements an advanced **Unsupervised Domain Adaptation (UDA)** pipeline for the Person Re-Identification (Person Re-ID) task. The goal is to adapt a model pre-trained on a source dataset (e.g., Market-1501) to perform effectively on a target dataset (e.g., DukeMTMC-reID) without using any of the target's labels.

The pipeline is built upon a **progressive pseudo-labeling** strategy using DBSCAN clustering, enhanced by an innovative **refinement method based on camera priors**.

## ✨ Key Features

-   **Progressive Pseudo-Labeling:** Utilizes the DBSCAN clustering algorithm to generate pseudo-labels at each epoch, with a confidence threshold that gradually relaxes to incorporate more data as training progresses.
-   **Camera Prior Refinement:** A label correction step where an image's neighbors vote on its identity. Votes from neighbors captured by a different camera are given higher weight, leveraging the prior that the same person is likely to appear across multiple cameras.
-   **Advanced Training Techniques:**
    -   **Triplet Loss** with temperature scheduling and hard mining.
    -   Identity-based sampler to create balanced mini-batches.
    -   Robust data augmentation, including `RandomErasing`.
-   **Modular and Configurable:** Fully scripted with `argparse` for easy hyperparameter tuning via the command line.
-   **VS Code Integration:** Includes a `launch.json` configuration for out-of-the-box debugging.
-   **Results Analysis:** Automatically generates and saves analysis plots to visualize performance and training dynamics.

## 🚀 Performance Highlights

The domain adaptation process dramatically improves the model's performance on the DukeMTMC-reID dataset. The **mAP improves from 3.26% to 43.81%** and the **Rank-1 accuracy from 7.85% to 64.09%**.

| ![Performance Comparison](output/graphs/2_comparaison_avant_apres.png) |
| :-------------------------------------------------------------------: |
| *Comparison of mAP and Rank-1 metrics before and after adaptation.*    |

| ![Performance Evolution](output/graphs/1_evolution_performances.png) |
| :-----------------------------------------------------------------: |
| *Evolution of mAP and Rank-1 accuracy over the adaptation epochs.*   |

## 📂 Project Structure

```
reid-project/
├── .vscode/
│   └── launch.json         # VS Code debugger configuration
├── data/
│   └── DukeMTMC-reID/      # Target dataset (to be downloaded)
├── output/
│   ├── graphs/             # Generated performance graphs
│   └── best_model_...pth   # Best model saved during training
├── pretrained_models/
│   └── Market1501_...pth   # Pre-trained model (to be downloaded)
├── venv/                   # Python virtual environment
├── main.py                 # Main script for training and evaluation
└── requirements.txt        # Python dependency list
```

## 🛠️ Getting Started

### Prerequisites

-   Python 3.8+ and `pip`
-   Git
-   (Recommended) An NVIDIA GPU with CUDA for optimal performance

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd reid-project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data and Models:**
    -   Download the **DukeMTMC-reID** dataset and extract it into the `data/` folder.
    -   Download the pre-trained model on **Market-1501** and place the `.pth` file in the `pretrained_models/` folder.
    -   Ensure your folder structure matches the one described above.

## 📈 Usage

### Running Training & Evaluation

Execute the main script from your terminal. The script will first evaluate the baseline model, then run the adaptation process, and finally evaluate the best-performing adapted model.

```bash
python main.py
```

The results, best model checkpoint, and analysis graphs will be saved in the `output/` directory.

### Using the VS Code Debugger

Thanks to the included `launch.json` file, you can easily debug the code:
1.  Open the `reid-project/` folder in VS Code.
2.  Open the `main.py` file.
3.  Set breakpoints as needed.
4.  Navigate to the "Run and Debug" view (Ctrl+Shift+D).
5.  Click the green "▶️" play button next to "Python: Lancer le script principal".

### Command-Line Arguments

You can easily modify hyperparameters without editing the source code.

```bash
# Example: Train for 10 epochs with a different learning rate
python main.py --adaptation_epochs 10 --adaptation_lr 1e-5

# Force the use of CPU
python main.py --device cpu

# See all available options
python main.py --help
```

## 💡 Future Work

-   Integrate other clustering algorithms (e.g., K-Means, HDBSCAN).
-   Test the pipeline on other target datasets (e.g., MSMT17).
-   Implement automated hyperparameter search (e.g., with Optuna).
-   Explore more advanced data augmentation techniques.
