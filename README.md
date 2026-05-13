# Clustering-Based Action Clip Classification

Project for **MTL782 Data Mining (Spring 2024-25, IIT Delhi)**. The goal is to
cluster short action clips into **15 clusters** and use those cluster labels to
classify new clips. The work is split into:

- **Non-competitive track**: restricted models and feature engineering.
- **Competitive track**: broader modeling freedom.

## Repository contents

- `CodeNC.ipynb` — Non-competitive pipeline (feature extraction + clustering).
- `CodeC.ipynb` — Competitive pipeline using precomputed ResNet50 features.
- `FeaturesC/` — Standardized ResNet50 features for the competitive track.
- `train_set/train_set/` — Training clips (class encoded in filenames).
- `test_set/test_set/` — Test clips.
- `Problem_Statement.pdf` — Full assignment statement.
- `report.pdf` — Detailed report and results.

## Method summary

- Sample frames from each clip and preprocess them.
- Extract features (e.g., HOG and ResNet50; additional features are explored in
  `CodeNC.ipynb`).
- Cluster training clips and assign labels based on cluster membership.
- Predict labels for test clips using the learned assignments.
- Competitive track leverages precomputed ResNet50 features and model selection.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

> **Note:** TensorFlow 1.15 and `dm-sonnet` are required for some non-competitive
> feature extractors.

## Usage

### Non-competitive track

Open `CodeNC.ipynb` in Jupyter and run the notebook end-to-end. Update paths if
you move the dataset directories.

### Competitive track

Open `CodeC.ipynb`. The notebook loads:

- `FeaturesC/resnet50_features_standardized_l2.csv`
- `FeaturesC/resnet50_test_features_standardized_l2.csv`

and trains/evaluates models on the provided features.

This code is adapted from my code originally written in kaggle. Click [here](https://www.kaggle.com/code/rudranilnaskar/noncomp-action-classification) for the original kaggle notebook.

## Results

See `report.pdf` for quantitative results, comparisons, and discussion.

## Notes

- Feature extraction over videos is compute-intensive; GPU acceleration helps.
- Clip filenames in `train_set/train_set/` encode their class labels.
