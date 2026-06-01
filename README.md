# 🌸 Iris Flower Classification with K-Nearest Neighbors (KNN)

> Classifying iris species using distance-based machine learning — a clean study in KNN, optimal K selection, and classification evaluation.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## 📌 Project Overview

The Iris dataset is a classic benchmark in classification — three flower species (Setosa, Versicolour, Virginica) described by four morphological features. This project implements a **K-Nearest Neighbors (KNN) classifier** to correctly identify species based on sepal and petal measurements.

Beyond basic classification, the project focuses on **finding the optimal K** through systematic comparison, understanding the bias-variance tradeoff, and demonstrating why **feature scaling is non-negotiable** for distance-based algorithms.

---

## ✨ Features

- 📊 **Pairplot EDA** — visualise class separability across all feature combinations
- 🧹 **Data Preprocessing** — missing value checks, stratified train/test split
- ⚖️ **Feature Standardisation** — StandardScaler applied before distance calculation
- 🔍 **Optimal K Selection** — train vs test accuracy plotted across K = 1 to 20
- 🤖 **KNN Training** — final model trained with best-performing K value
- 📋 **Full Evaluation** — accuracy, precision, recall, F1-score per class
- 🟦 **Confusion Matrix** — visual breakdown of correct vs misclassified predictions
- 📊 **K Comparison Table** — performance summary across 8 different K values

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Manipulation | pandas, NumPy |
| Machine Learning | scikit-learn |
| Visualisation | matplotlib, seaborn |
| Environment | Jupyter Notebook / VS Code |
| Dataset | Iris (sklearn built-in) |

---

## 📸 Screenshots

### Pairplot — Feature Separability by Species
![Pairplot](screenshots/pairplot.png)
> Setosa is clearly linearly separable. Versicolour and Virginica overlap slightly — this is where K matters.

### Accuracy vs K Value
![K Selection](screenshots/accuracy_vs_k.png)
> Train and test accuracy plotted across K=1–20. K=1 overfits; the optimal range is typically K=5–7.

### Confusion Matrix
![Confusion Matrix](screenshots/confusion_matrix.png)
> Near-perfect classification. Any misclassifications occur between Versicolour and Virginica.

### Classification Report
![Classification Report](screenshots/classification_report.png)
> Precision, recall, and F1-score per class — all close to 1.00.

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/fsafva13-coder/iris-knn-classifier.git
cd iris-knn-classifier

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook knn_classifier_task3.ipynb
```

> **VS Code users:** Open the `.ipynb` file directly with the Jupyter extension installed.

---

## 🚀 Usage

1. Open `knn_classifier_task3.ipynb`
2. Run all cells sequentially (`Run All` or `Shift + Enter`)
3. The notebook will:
   - Load the Iris dataset automatically (no download needed)
   - Preprocess, scale, and split the data
   - Plot the accuracy curve across K=1–20
   - Train the final model with the optimal K
   - Print the classification report and display all visualisations

**Expected output:**
```
Best K: 5 | Test Accuracy: 1.0000

Classification Report:
              precision    recall  f1-score
    Setosa       1.00      1.00      1.00
Versicolour      1.00      1.00      1.00
  Virginica      1.00      1.00      1.00
```

---

## 📁 Project Structure

```
iris-knn-classifier/
│
├── knn_classifier_task3.ipynb   # Main notebook (EDA → Training → Evaluation)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
└── screenshots/                 # Visualisation outputs
    ├── pairplot.png
    ├── accuracy_vs_k.png
    ├── confusion_matrix.png
    └── classification_report.png
```

---

## 📊 Results & Performance

| Metric | Value |
|--------|-------|
| **Best K** | 5 |
| **Test Accuracy** | 100% |
| **Precision (avg)** | 1.00 |
| **Recall (avg)** | 1.00 |
| **F1-Score (avg)** | 1.00 |

### K Value Comparison

| K | Test Accuracy | Correct | Wrong |
|---|--------------|---------|-------|
| 1 | 0.9667 | 29/30 | 1 |
| 3 | 1.0000 | 30/30 | 0 |
| 5 | 1.0000 | 30/30 | 0 |
| 7 | 1.0000 | 30/30 | 0 |
| 9 | 1.0000 | 30/30 | 0 |
| 11 | 0.9667 | 29/30 | 1 |
| 15 | 0.9667 | 29/30 | 1 |
| 20 | 0.9667 | 29/30 | 1 |

**Key Findings:**

- 🌸 **Setosa** is perfectly separable from the other two species across all K values — confirmed visually by the pairplot
- 🔀 **Versicolour vs Virginica** is the harder boundary — small K values are more sensitive to this overlap
- ⚠️ **K=1 overfits** — it memorises training points and is sensitive to noise
- 📏 **Feature scaling is non-negotiable** — without StandardScaler, petal length dominates the distance metric and distorts every prediction

---

## 🧩 Challenges & Learnings

- **Scaling before KNN is essential, not optional.** Initial runs without StandardScaler produced inconsistent K curves because features with larger absolute ranges (petal length) dominated the Euclidean distance calculation entirely. One preprocessing step changed the results significantly.
- **K=1 always achieves 100% training accuracy** — a textbook overfitting case. The train vs test accuracy plot makes this bias-variance tradeoff immediately visual and easy to explain.
- **The Iris dataset is deceptively clean** — real-world classification tasks will have overlapping classes, imbalanced data, and noisy features. This project is the right foundation before tackling those complexities.

---

## 🔮 Future Improvements

- [ ] Test on a more challenging, real-world dataset (e.g. breast cancer, credit default)
- [ ] Add **cross-validation** (K-Fold) for more robust K selection
- [ ] Implement **weighted KNN** (distance-weighted voting) and compare
- [ ] Visualise the **decision boundary** using PCA to reduce to 2D
- [ ] Compare KNN against **SVM and Random Forest** on the same dataset
- [ ] Build an interactive **Streamlit app** for live species prediction

---

## 🔗 Demo / Live Link

> 📓 Notebook available in this repository — clone and run locally.  
> 🚀 Streamlit deployment: *coming soon*

---

## 👩‍💻 Author

**Fathima Safva**  
BSc Computer Science | University of West London (UAE Campus)  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/fathima-safva-578294315)
[![GitHub](https://img.shields.io/badge/GitHub-fsafva13--coder-black?style=flat&logo=github)](https://github.com/fsafva13-coder)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*Built as part of the Codveda Technologies Machine Learning Internship — Level 1, Task 3.*
