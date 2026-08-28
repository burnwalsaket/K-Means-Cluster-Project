# University Clustering with K-Means

Using unsupervised K-Means clustering to group U.S. colleges into two clusters based on institutional features, then checking how well the clusters align with the true Private vs. Public label — without ever showing the model that label during training.

## Overview

This project explores whether K-Means, given no labels at all, can naturally rediscover the Private vs. Public distinction between colleges purely from structural features like tuition, enrollment, and admissions statistics. The true `Private` label is only used afterward, to evaluate how well the unsupervised clusters line up with reality — a common way to sanity-check clustering when ground truth happens to be available.

## Dataset

- **Source:** `College_Data` — U.S. News college statistics
- **Size:** 777 universities, 18 features (17 used for clustering)
- **Label (held out during clustering):** `Private` (Yes/No)
- **Features used:** `Apps`, `Accept`, `Enroll`, `Top10perc`, `Top25perc`, `F.Undergrad`, `P.Undergrad`, `Outstate`, `Room.Board`, `Books`, `Personal`, `PhD`, `Terminal`, `S.F.Ratio`, `perc.alumni`, `Expend`, `Grad.Rate`

## Data Cleaning

- Found and corrected a data entry error: Cazenovia College had a `Grad.Rate` above 100%, which was capped at 100

## Exploratory Data Analysis

- Scatterplot of `Grad.Rate` vs. `Room.Board`, colored by Private/Public
- Scatterplot of `F.Undergrad` vs. `Outstate` tuition, colored by Private/Public
- Overlaid histograms (via `FacetGrid`) of `Outstate` tuition and `Grad.Rate`, split by Private/Public, to visually confirm separability before clustering

Private and public schools show visibly different distributions in out-of-state tuition and full-time undergraduate enrollment, which is what the clustering step attempts to exploit without being told the label directly.

## Methodology

1. **Model:** `KMeans(n_clusters=2)` from scikit-learn, fit on all 17 numeric features with the `Private` column dropped
2. **Evaluation:** Since ground-truth labels exist for this exercise (not typical for real unsupervised problems), the true `Private` label was encoded as a binary `Cluster` column and compared against `kmeans.labels_` using a confusion matrix and classification report

## Results

Raw comparison against `kmeans.labels_`:

| Metric | Value |
|---|---|
| Accuracy (raw cluster IDs) | 22% |

**Important caveat:** K-Means cluster IDs (0 and 1) are arbitrary — the algorithm has no notion of which cluster "means" private vs. public, so it may assign them opposite to how the true labels are encoded. Relabeling the clusters to their best matching orientation (i.e., treating cluster 0 as Private and cluster 1 as Public instead) gives:

| Metric | Value |
|---|---|
| Accuracy (best label alignment) | ~78% |

This is a good illustration of a common pitfall when validating clustering against known labels: always check both possible label orientations before concluding a clustering algorithm "failed." With that correction, K-Means recovers the Private/Public split reasonably well using only structural features, with no access to the label itself during training.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, Matplotlib, Seaborn, scikit-learn (`KMeans`, `classification_report`, `confusion_matrix`)

## Repository Structure

```
├── K_Means_Clustering_Project.ipynb   # Full analysis notebook
├── College_Data                        # Dataset
└── README.md
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook K_Means_Clustering_Project.ipynb
```
