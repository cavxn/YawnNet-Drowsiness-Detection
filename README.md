# YawnNet: Driver Drowsiness Detection using Deep Learning

## Overview

**YawnNet** is a deep learning-based computer vision system designed to detect driver yawning as an indicator of fatigue and drowsiness. The project uses a Convolutional Neural Network (CNN) trained on facial images to classify whether a driver is yawning or not. Detecting yawning is an important step in identifying driver fatigue and improving road safety.

This system can be integrated into driver monitoring systems to alert drivers before fatigue becomes dangerous.

---

## Features

* Deep learning-based **yawn detection using CNN**
* Binary classification: **Yawn vs No Yawn**
* Dataset preprocessing and augmentation
* Comprehensive **model evaluation and visualization**
* Multiple performance metrics and graphs
* Misclassification analysis
* Feature visualization using **t-SNE**

---

## Dataset

The dataset contains images of faces labeled into two categories:

* **Yawn**
* **No Yawn**

Directory structure:

```
yawn_dataset/
│
├── train/
│   ├── yawn/
│   └── no_yawn/
│
└── val/
    ├── yawn/
    └── no_yawn/
```

⚠️ Dataset is not included in this repository due to size limitations.

---

## Model Architecture

The model is a **Convolutional Neural Network (CNN)** trained to classify facial images.

Typical architecture components:

* Convolutional layers
* ReLU activations
* MaxPooling layers
* Fully connected layers
* Sigmoid output layer for binary classification

The model is trained to detect visual patterns associated with yawning, particularly around the mouth region.

---

## Project Structure

```
YawnNet-Drowsiness-Detection/
│
├── yawn_model_analysis.ipynb
├── yawn_detection_model4.keras
├── .gitignore
├── README.md
│
└── yawn_dataset/ (ignored in git)
    ├── train/
    └── val/
```

---

## Evaluation Metrics

The model is evaluated using several metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curve
* Precision-Recall Curve
* Confusion Matrix

These metrics help evaluate classification performance and model reliability.

---

## Visualization & Graphs

The evaluation notebook generates multiple graphs including:

* Dataset class distribution
* Sample dataset images
* Confusion matrix
* ROC curve
* Precision–Recall curve
* Prediction confidence distribution
* Prediction probability timeline
* Misclassification gallery
* t-SNE feature embeddings

These visualizations help analyze model performance and behavior.

---

## Installation

Clone the repository:

```
git clone https://github.com/cavxn/YawnNet-Drowsiness-Detection.git
cd YawnNet-Drowsiness-Detection
```

Install dependencies:

```
pip install tensorflow
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

---

## Usage

Run the evaluation notebook:

```
jupyter notebook yawn_model_analysis.ipynb
```

The notebook will:

1. Load the trained model
2. Load the dataset
3. Run predictions
4. Generate evaluation metrics
5. Produce analysis graphs

---

## Applications

This project can be applied in:

* Driver monitoring systems
* Smart vehicle safety systems
* Driver fatigue detection
* Automotive AI safety features

---

## Future Improvements

Possible improvements for the project include:

* Real-time webcam-based yawning detection
* Integration with eye-blink detection
* Real-time driver fatigue score
* Mobile or embedded deployment
* Integration into full driver monitoring systems

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Author

Cavin S

---

## License

This project is intended for educational and research purposes.
