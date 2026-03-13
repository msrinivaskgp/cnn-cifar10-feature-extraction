# CIFAR10 CNN Feature Extraction

## Overview

This repository implements a **10-layer Convolutional Neural Network (CNN)** using **PyTorch** for image classification on the **CIFAR-10 dataset**.
The model is trained using stochastic gradient descent with learning rate scheduling and early stopping.

After training, **feature representations are extracted from each convolutional layer** of the CNN.
These features are spatially averaged and saved as **CSV files**, which can later be used for further analysis, such as clustering, classification, or probabilistic modeling.

This project is useful for studying **layer-wise feature representations in deep neural networks**.

---

## Dataset

The model is trained and evaluated on the **CIFAR-10 dataset**, which contains:

* **60,000 color images**
* Image size: **32 × 32**
* **10 classes**

Classes include:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

The dataset is automatically downloaded when the code is executed.

---

## Model Architecture

The implemented network consists of:

* **10 convolutional layers**
* **Batch normalization**
* **ReLU activations**
* **Fully connected classification layers**

Key characteristics:

* No padding in convolution layers
* Kernel size: **3 × 3**
* Progressive increase in feature channels
* Dropout used in the fully connected layer

---

## Training Details

| Parameter               | Value  |
| ----------------------- | ------ |
| Optimizer               | SGD    |
| Learning rate           | 0.001  |
| Momentum                | 0.9    |
| Weight decay            | 1e-4   |
| Epochs                  | 40     |
| Batch size              | 128    |
| Learning rate scheduler | StepLR |
| Early stopping patience | 5      |

Data augmentation used:

* Random horizontal flip
* Random rotation

---

## Feature Extraction

After training, the model extracts features from each convolution layer.

For each feature map:

1. The spatial dimensions are flattened.
2. The mean value of each channel is computed.
3. The resulting feature vector represents the layer output.

These features are saved as CSV files for further analysis.

---

## Project Structure

```
cnn-cifar10-feature-extraction
│
├── data/
│   (CIFAR10 dataset downloaded automatically)
│
├── outputs/
│   ├── features/
│   │   ├── L1_pn.csv
│   │   ├── L2_pn.csv
│   │   ├── L3_pn.csv
│   │   ├── L4_pn.csv
│   │   ├── L5_pn.csv
│   │   ├── L6_pn.csv
│   │   ├── L7_pn.csv
│   │   ├── L8_pn.csv
│   │   ├── L9_pn.csv
│   │   ├── L10_pn.csv
│   │   └── labels_pn.csv
│   │
│   └── models/
│       └── best_model.pth
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── feature_extraction.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/cnn-cifar10-feature-extraction.git
cd cnn-cifar10-feature-extraction
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the complete pipeline:

```bash
python main.py
```

The script will:

1. Download CIFAR10 dataset
2. Train the CNN model
3. Evaluate performance
4. Extract features from all convolution layers
5. Save features to CSV files

---

## Outputs

After execution, the following outputs are generated:

### Model

```
outputs/models/best_model.pth
```

### Extracted Features

```
outputs/features/
```

Layer-wise feature files:

* L1_pn.csv
* L2_pn.csv
* L3_pn.csv
* L4_pn.csv
* L5_pn.csv
* L6_pn.csv
* L7_pn.csv
* L8_pn.csv
* L9_pn.csv
* L10_pn.csv

Label file:

* labels_pn.csv

---

## Applications

The extracted CNN features can be used for:

* Feature analysis
* Dimensionality reduction
* Clustering
* Probabilistic modeling
* Gaussian Mixture Models
* Decision tree learning
* Explainable AI research

---

## Requirements

The project requires:

* Python 3.8+
* PyTorch
* Torchvision
* NumPy
* Pandas
* tqdm

---

## License

This project is intended for **research and educational purposes**.

---

## Author

**M Srinivas**

PhD Research Scholar
Indian Institute of Technology Kharagpur
