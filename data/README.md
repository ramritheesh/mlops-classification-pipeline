# Dataset Information: Breast Cancer Wisconsin (Diagnostic)

## Source
- **Origin**: UCI Machine Learning Repository / `sklearn.datasets.load_breast_cancer`
- **Donor**: Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (University of Wisconsin)
- **License**: Public Domain / BSD-3-Clause via scikit-learn

## Characteristics
- **Instances**: 569 samples
- **Features**: 30 numeric continuous features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass describing characteristics of cell nuclei:
  - Radius, Texture, Perimeter, Area, Smoothness
  - Compactness, Concavity, Concave points, Symmetry, Fractal dimension
  - Captured as `mean`, `standard error (se)`, and `worst` (largest) for each property
- **Target**: Binary classification
  - `0`: Malignant
  - `1`: Benign

## Crucial Disclaimer
> **IMPORTANT NOTE**: This dataset is used strictly as an educational benchmark for demonstration of MLOps software engineering practices (data validation, experiment tracking, model registry, FastAPI serving, containerization, and monitoring).
> 
> **THIS MODEL AND PIPELINE ARE NOT CLINICAL DIAGNOSTIC TOOLS AND MUST NEVER BE USED FOR REAL-WORLD MEDICAL DIAGNOSIS OR CLINICAL DECISION-MAKING.**
