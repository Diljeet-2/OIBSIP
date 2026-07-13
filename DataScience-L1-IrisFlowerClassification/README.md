# 🌸 Iris Flower Classification

An end-to-end Machine Learning project to classify the species of Iris flowers (**Setosa**, **Versicolor**, and **Virginica**) using their physical measurements: sepal length, sepal width, petal length, and petal width.

Built with **Python**, **scikit-learn**, **Pandas**, **Seaborn**, and **Matplotlib**, this repository contains a complete analysis pipeline and a clean, beautifully formatted Jupyter Notebook.

---

## 🚀 Features & Deliverables

- [x] **Data Ingestion**: Programmatic loading of the Iris dataset directly via `sklearn.datasets`.
- [x] **Exploratory Data Analysis (EDA)**: Descriptive statistics table, shapes, types, and NULL value verifications.
- [x] **Premium Visualizations**: Feature distributions via multi-dimensional pairplots, detailed box plots, and correlation matrices.
- [x] **Feature Selection Discussion**: In-depth analysis of feature correlations and class separability.
- [x] **Machine Learning Pipeline**: Clean train/test splitting (80/20 ratio, stratified) and model training.
- [x] **Multi-Model Comparison**: Evaluated four different classifiers:
  1. **Logistic Regression** (Linear Baseline)
  2. **K-Nearest Neighbors** (Instance-based)
  3. **Decision Tree** (Non-linear Hierarchical)
  4. **Random Forest** (Ensemble Bagging)
- [x] **Evaluation Metrics**: Computed accuracy, precision, recall, and F1-scores, supplemented by inline visual confusion matrices.
- [x] **Model Justification**: Logical reasoning selecting the most lightweight, robust model.

---

## 📊 Exploratory Data Analysis & Visualizations

Visual files are saved in the [screenshots/](screenshots/) directory.

### 1. Pairwise Feature Distributions
This diagonal kernel density and scatter plot grid displays pairwise relationships between the four measurements, color-coded by Iris species. Setosa is completely linearly separable, while Versicolor and Virginica present minor overlap in two-dimensional projections.

![Iris Pairplot](screenshots/iris_pairplot.png)

### 2. Feature Box Plots & Stripplots
Box plots with overlaid scatter strip plots show the distribution range, median, quartiles, and density of each feature. Petal measurements present the highest variation between species and the smallest overlap, making them highly discriminative.

![Iris Boxplots](screenshots/iris_boxplots.png)

### 3. Feature Correlations
Strong Pearson correlation coefficients characterize relationships between Petal Length, Petal Width, and Sepal Length, whereas Sepal Width displays weak, negative correlation trends.

![Feature Correlation Matrix](screenshots/iris_correlation.png)

---

## 🧠 Feature Selection Discussion

By reviewing the visualizations and statistical metrics, we identify the following discriminative features:

* **Petal Length & Petal Width (Highest Discriminative Power)**: 
  * `setosa` has visual boundaries completely disjointed from other classes (Petal Length $< 2.0$ cm and Petal Width $< 0.7$ cm).
  * `versicolor` and `virginica` display their highest separation boundaries along the petal length/width axis.
* **Sepal Dimensions (Moderate to Low Discriminative Power)**: 
  * Sepal length and sepal width exhibit high variance overlay between `versicolor` and `virginica`, suggesting that models relying purely on sepal dimensions would suffer from low accuracy.

---

## 🤖 Model Evaluation & Comparison

Here is the classification performance of our trained models on the test set:

| Model | Test Accuracy | Precision | Recall | F1-Score | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **100.00%** | **1.00** | **1.00** | **1.00** | **Best Choice** |
| **K-Nearest Neighbors (k=5)** | **100.00%** | **1.00** | **1.00** | **1.00** | Highly Robust |
| **Random Forest (n=100)** | **100.00%** | **1.00** | **1.00** | **1.00** | Powerful Ensemble |
| **Decision Tree** | **96.67%** | **0.97** | **0.97** | **0.97** | Minor Overfitting |

### Confusion Matrices
Below are the confusion matrices side-by-side highlighting classifications. Only the Decision Tree misclassified a single Versicolor/Virginica sample (attributing Versicolor as Virginica).

![Confusion Matrices](screenshots/confusion_matrices.png)

---

## 🏆 Declared Best Model & Justification

**Selected Best Model: Logistic Regression**

### Justification:
1. **Perfect Performance**: Achieved **100.00% test accuracy** with perfect precision, recall, and F1-scores across all three species.
2. **Occam's Razor**: It is the simplest model tested. On small datasets ($N=150$) with linearly separable attributes, simpler models are less prone to overfitting than complex ensembles (e.g., Random Forest) or non-linear trees.
3. **Interpretability**: Logistic Regression outputs class log-probabilities directly. The model coefficients can be parsed to understand how each feature shifts the odds of classification.
4. **Efficiency**: It is computationally lightweight, requiring negligible memory and CPU cycles during both training and high-throughput inference.

---

## 📁 Repository Structure

```tree
Oasis 1/
├── app.py                              # Interactive Streamlit Web Application Dashboard
├── Iris_Flower_Classification.ipynb    # Clean, step-by-step Jupyter Notebook
├── README.md                           # Professional presentation of results (This file)
├── run_analysis.py                     # Python automation pipeline script
└── screenshots/                        # Folder containing plots and data tables
    ├── confusion_matrices.png          # Side-by-side model confusion matrices
    ├── descriptive_statistics.csv      # Tabular summary statistics (CSV format)
    ├── iris_boxplots.png               # Sepal/Petal box & stripplots per species
    ├── iris_correlation.png            # Pearson correlation heatmap
    └── iris_pairplot.png               # Custom-themed pairplot matrix
```

---

## 💻 How to Run

1. **Prerequisites**: Make sure Python 3.x is installed. Install dependencies using:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib plotly streamlit notebook
   ```
2. **Running the Streamlit Interactive Dashboard**:
   Launch the web dashboard to play with inputs and visualize models in real-time:
   ```bash
   streamlit run app.py
   ```
3. **Viewing & Running the Notebook**:
   Launch Jupyter Notebook in your workspace or open `Iris_Flower_Classification.ipynb` directly using VS Code (with the Jupyter Extension). Run all cells to execute analysis and witness plots inline.
   ```bash
   jupyter notebook
   ```
4. **Re-generating Visualizations Programmatically**:
   Run the analysis script to re-generate CSV summaries and image files:
   ```bash
   python run_analysis.py
   ```
