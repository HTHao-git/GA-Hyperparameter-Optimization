# GA-Hyperparameter-Optimization

**Genetic Algorithm Framework for Machine Learning Hyperparameter Tuning**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

> A comprehensive framework for optimizing machine learning model hyperparameters using Genetic Algorithms with advanced operators, multi-objective optimization, and complete analysis tools.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Supported Models](#supported-models)
- [Datasets](#datasets)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Results & Analysis](#results--analysis)
- [Research Findings](#research-findings)
- [Performance Benchmarks](#performance-benchmarks)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Configuration](#configuration)
- [Custom Datasets](docs/CUSTOM_DATASETS.md)  <!-- ADD -->
- [Advanced Configuration](docs/CONFIGURATION_GUIDE.md)  <!-- ADD -->
- [Results & Analysis](#results--analysis)

---

## Overview

This project implements a **production-ready Genetic Algorithm framework** for automated hyperparameter optimization across multiple machine learning models. Built for research and thesis work, it provides:

- **Advanced GA operators** (selection, crossover, mutation)
- **Multi-objective optimization** (NSGA-II)
- **Adaptive parameters** (diversity-based, fitness-based)
- **Comprehensive analysis** (statistical tests, visualizations, reports)
- **5 ML models** optimized out-of-the-box
- **4 benchmark datasets** included

### **Why This Project?**

 **Research-Grade:** Publication-ready results with statistical rigor  
 **Modular:** Easy to extend with new models/datasets  
 **Complete:** From data loading to HTML reports  
 **Documented:** ~10,000 lines with comprehensive comments  
 **Validated:** Tested on imbalanced, high-dimensional datasets
 **Some Background thoughts**: While I was researching for the my graduation thesis, I noticed that ML inherently has a problem of having to understand both the dataset quirks and the models attributes in order to steer the learning model into the desired direction. This is why HPO algorithms exist, to help explore and potentially find the best configuration that aligns with our expected output.  

---

## Features

### **Genetic Algorithm Core**

- **8 Selection Methods:** Tournament, Roulette, Rank, SUS, Elitism, Boltzmann, Truncation, Random
- **6 Crossover Methods:** Single-point, Two-point, Uniform, Arithmetic, BLX-α, SBX
- **6 Mutation Methods:** Uniform, Gaussian, Polynomial, Adaptive, Boundary, Non-uniform
- **Multi-Objective:** NSGA-II implementation with Pareto front optimization
- **Diversity Maintenance:** Fitness sharing, crowding distance, niching
- **Adaptive Parameters:** Dynamic mutation/crossover rates based on diversity/fitness

### **Machine Learning Models**

| Model | Library | Optimized Parameters |
|-------|---------|---------------------|
| **Random Forest** | scikit-learn | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features |
| **XGBoost** | xgboost | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda, gamma |
| **LightGBM** | lightgbm | n_estimators, max_depth, learning_rate, num_leaves, subsample, reg_alpha, reg_lambda |
| **Neural Network** | scikit-learn | Architecture (layers), learning_rate, alpha, solver, activation, batch_size |
| **SVM** | scikit-learn | C, kernel, gamma, degree, coef0, class_weight |

### **Data Pipeline**

- **Missing Value Handling:** Mean, median, forward-fill, KNN imputation
- **Feature Scaling:** Standard, MinMax, Robust scalers
- **Class Balancing:** SMOTE, ADASYN, Random Over/Under sampling
- **Dimensionality Reduction:** PCA with variance retention
- **4 Benchmark Datasets:** Auto-download via Kaggle API

### **Analysis & Reporting**

- **Metrics:** Accuracy, Precision, Recall, F1, Balanced Accuracy, ROC-AUC
- **Statistical Tests:** Paired t-test, Wilcoxon signed-rank, Cohen's d
- **Visualizations:** Convergence plots, diversity charts, confusion matrices, radar charts
- **Reports:** HTML (interactive), Text, JSON formats
- **Timing Tracking:** Optimization time, evaluation time per model

---

## Quick Start

### **1. Clone Repository**

```bash
git clone https://github.com/HTHao-git/GA-Hyperparameter-Optimization.git
cd GA-Hyperparameter-Optimization
```

### **2. Setup Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **3. Configure Kaggle API** (for datasets)

```bash
# Download kaggle. json from https://www.kaggle.com/settings
# Place in: 
#   Windows: C:\Users\<YourUsername>\.kaggle\kaggle.json
#   Linux/Mac: ~/.kaggle/kaggle.json

# Set permissions (Linux/Mac)
chmod 600 ~/. kaggle/kaggle.json
```

### **4. Run Your First Optimization**

```python
from preprocessing. data_loader import DatasetLoader
from ga. ml_optimizer import MLOptimizer
from ga.genetic_algorithm import GAConfig

# Load dataset
loader = DatasetLoader(interactive=False)
X, y, _ = loader.load_dataset('secom')

# Configure GA
ga_config = GAConfig(
    population_size=20,
    num_generations=15,
    mutation_rate=0.2,
    crossover_rate=0.8,
    adaptive_mutation=True
)

# Optimize Random Forest
optimizer = MLOptimizer(X, y, model_type='random_forest', ga_config=ga_config)
results = optimizer.optimize()

print(f"Best CV Score: {results['cv_score']:.4f}")
print(f"Test Score: {results['test_score']:.4f}")
print(f"Best Config: {results['config']}")
```

### **5. Compare Multiple Models**

```bash
python compare_models.py
```

**Output:** `outputs/model_comparison/` with HTML report, plots, statistical tests

---

## 📁 Project Structure

```
GA-Hyperparameter-Optimization/
│
├── ga/                           # Genetic Algorithm Core
│   ├── genetic_algorithm.py      # Main GA engine
│   ├── selection.py              # 7 selection methods
│   ├── crossover.py              # 6 crossover operators
│   ├── mutation. py               # 6 mutation operators
│   ├── multi_objective.py        # NSGA-II implementation
│   ├── diversity.py              # Diversity maintenance
│   ├── adaptive. py               # Adaptive parameter control
│   └── types.py                  # Data structures (Individual, GAConfig)
│
├── models/                       # ML Model Optimizers
│   ├── xgboost_optimizer.py      # XGBoost optimization
│   ├── lightgbm_optimizer.py     # LightGBM optimization
│   ├── neural_network_optimizer.py  # Neural network architecture search
│   └── svm_optimizer.py          # SVM kernel/parameter optimization
│
├── preprocessing/                # Data Processing Pipeline
│   ├── data_loader.py            # Dataset loading & caching
│   ├── missing_values.py         # Missing value imputation
│   ├── scaling.py                # Feature scaling methods
│   ├── smote_handler.py          # Class balancing (SMOTE/ADASYN)
│   └── pca.py                    # PCA dimensionality reduction
│
├── utils/                        # Utility Functions
│   ├── logger.py                 # Logging system
│   ├── colors.py                 # Terminal color formatting
│   ├── metrics.py                # Metrics calculation
│   ├── visualization.py          # Plotting functions
│   ├── statistical_tests.py      # Statistical significance tests
│   └── report_generator.py       # HTML/Text report generation
│
├── datasets/                     # Downloaded datasets (auto-created)
├── outputs/                      # Results & reports (auto-created)
│
├── compare_models.py             # Multi-model comparison script
├── compare_fitness_metrics.py    # Fitness metric analysis (Accuracy vs F1)
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

---

## Supported Models

### **1. Random Forest**

```python
from ga.ml_optimizer import MLOptimizer

optimizer = MLOptimizer(X, y, model_type='random_forest')
results = optimizer.optimize()
```

**Hyperparameters Optimized:**
- `n_estimators`: Number of trees
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split
- `min_samples_leaf`: Minimum samples per leaf
- `max_features`: Features per split

---

### **2. XGBoost**

```python
from models.xgboost_optimizer import XGBoostOptimizer

optimizer = XGBoostOptimizer(X, y)
results = optimizer.optimize()
```

**Hyperparameters Optimized:**
- `learning_rate`: Step size shrinkage
- `max_depth`: Tree depth
- `subsample`: Row sampling ratio
- `colsample_bytree`: Column sampling ratio
- `reg_alpha`, `reg_lambda`: L1/L2 regularization

---

### **3. LightGBM**

```python
from models.lightgbm_optimizer import LightGBMOptimizer

optimizer = LightGBMOptimizer(X, y)
results = optimizer.optimize()
```

**Hyperparameters Optimized:**
- `num_leaves`: Max leaves per tree
- `learning_rate`: Boosting learning rate
- `subsample_freq`: Bagging frequency
- Plus regularization parameters

---

### **4. Neural Networks (MLP)**

```python
from models.neural_network_optimizer import NeuralNetworkOptimizer

optimizer = NeuralNetworkOptimizer(X, y)
results = optimizer.optimize()
```

**Hyperparameters Optimized:**
- Architecture:  Number of layers & neurons
- `learning_rate_init`: Initial learning rate
- `alpha`: L2 regularization
- `solver`: Optimizer (Adam, SGD)
- `activation`: Activation function

**Example Architecture Found:** `[128, 64]` (2 hidden layers)

---

### **5. SVM**

```python
from models.svm_optimizer import SVMOptimizer

optimizer = SVMOptimizer(X, y)
results = optimizer.optimize()
```

**Hyperparameters Optimized:**
- `C`: Regularization parameter
- `kernel`: Kernel type (linear, RBF, poly, sigmoid)
- `gamma`: Kernel coefficient
- `degree`: Polynomial degree (for poly kernel)

---

## Datasets

| Dataset | Samples | Features | Classes | Domain | Imbalance |
|---------|---------|----------|---------|--------|-----------|
| **SECOM** | 1,567 | 590 | 2 | Semiconductor Manufacturing | 93.4% / 6.6% |
| **Fashion-MNIST** | 70,000 | 784 | 10 | Fashion Item Recognition | Balanced |
| **Isolet** | 7,797 | 617 | 26 | Speech Recognition (Letters) | Balanced |
| **Steel Plates** | 1,941 | 27 | 7 | Steel Defect Classification | Imbalanced |

**Auto-download via Kaggle API** - datasets cached in `datasets/` folder. 

---

## Usage Examples

### **Example 1: Basic Optimization**

```python
from preprocessing.data_loader import DatasetLoader
from models.xgboost_optimizer import XGBoostOptimizer
from ga.genetic_algorithm import GAConfig

# Load data
loader = DatasetLoader()
X, y, _ = loader.load_dataset('secom')

# Configure GA
config = GAConfig(
    population_size=20,
    num_generations=15,
    verbose=1
)

# Optimize
optimizer = XGBoostOptimizer(X, y, ga_config=config)
results = optimizer.optimize()

# Save results
optimizer.save_results('outputs/xgboost_results. json')
```

---

### **Example 2: Compare Models with Statistical Tests**

```python
from compare_models import main

# Runs RF vs XGBoost comparison with: 
# - Cross-validation
# - Statistical significance tests
# - Convergence visualizations
# - HTML report generation

main()
```

**Output:**
- `outputs/model_comparison/rf_convergence.png`
- `outputs/model_comparison/comparison_bars.png`
- `outputs/model_comparison/GA_Model_Comparison_report.html`

---

### **Example 3: Advanced GA Configuration**

```python
config = GAConfig(
    population_size=30,
    num_generations=25,
    crossover_rate=0.8,
    mutation_rate=0.2,
    elitism_rate=0.1,
    
    # Advanced features
    mutation_method='adaptive',      # Adaptive mutation
    mutation_strength='large',       # Exploration strength
    adaptive_mutation=True,          # Enable rate adaptation
    adaptive_method='diversity_based',  # Adapt based on diversity
    
    # Early stopping
    early_stopping=True,
    patience=8,
    diversity_threshold=0.0,  # Disable diversity-based stopping
    
    # Logging
    verbose=2  # Detailed logging
)
```

---

### **Example 4: Fitness Metric Comparison**

```bash
# Compare Accuracy vs Macro F1-score on imbalanced data
python compare_fitness_metrics. py
```

**Demonstrates:**
- Why accuracy fails on imbalanced data
- Macro F1 vs Weighted F1
- Impact on minority class detection

---

## Configuration

### **GA Configuration Options**

```python
GAConfig(
    # Population
    population_size=20,          # Number of individuals
    num_generations=15,          # Maximum generations
    
    # Operators
    crossover_rate=0.8,          # Crossover probability
    mutation_rate=0.2,           # Mutation probability
    elitism_rate=0.1,            # Elite preservation rate
    
    # Advanced Mutation
    mutation_method='adaptive',  # 'uniform', 'gaussian', 'polynomial', 'adaptive'
    mutation_strength='medium',  # 'small', 'medium', 'large'
    
    # Adaptive Parameters
    adaptive_mutation=True,      # Enable adaptive rates
    adaptive_method='diversity_based',  # 'diversity_based', 'fitness_based', 'schedule'
    
    # Early Stopping
    early_stopping=True,         # Enable early stopping
    patience=5,                  # Generations without improvement
    diversity_threshold=0.0,     # Min diversity (0.0 = disabled)
    
    # Performance
    cache_fitness=False,         # Cache fitness evaluations
    
    # Logging
    verbose=1,                   # 0=silent, 1=normal, 2=detailed
    random_state=42              # Reproducibility
)
```

---

## Results & Analysis

### **Metrics Calculated**

- **Classification:** Accuracy, Precision, Recall, F1-Score
- **Imbalanced Data:** Balanced Accuracy, Per-class metrics
- **Probabilistic:** ROC-AUC (when applicable)
- **Confusion Matrix:** Full breakdown

### **Statistical Tests**

- **Paired t-test:** Parametric comparison
- **Wilcoxon signed-rank:** Non-parametric alternative
- **Cohen's d:** Effect size measurement
- **Significance level:** α = 0.05

### **Visualizations Generated**

1. **Convergence Plot:** Fitness over generations
2. **Diversity Plot:** Population diversity tracking
3. **Comparison Bar Chart:** Multi-model performance
4. **Radar Chart:** Multi-metric comparison
5. **Confusion Matrix:** Classification breakdown

### **Reports**

- **HTML Report:** Interactive, publication-ready
- **Text Summary:** Quick reference
- **JSON Results:** Machine-readable for further analysis

**Example HTML Report Sections:**
- Experiment overview
- Dataset statistics
- Model configurations
- Performance metrics table
- Statistical test results
- Embedded visualizations

---

## Research Findings

### **1. Fitness Metric Selection on Imbalanced Data**

**Problem:** Using accuracy as fitness on imbalanced datasets (e.g., SECOM 93.4% / 6.6%) leads to degenerate solutions.

**Finding:**
- **Accuracy fitness:** Model predicts all majority class → 93.3% accuracy, 0% minority recall
- **Weighted F1 fitness:** Still biased (0.93 x F1 + 0.07 x F1₁) → Similar failure
- **Macro F1 fitness:** Treats classes equally → 85-90% accuracy, 40-70% minority recall

**Implication:** Macro-averaged metrics essential for imbalanced classification with GA.

---

### **2. Preprocessing Dominance**

**Finding:** On SECOM dataset, all 5 models achieved identical performance (93.36% CV, 93.31% test).

**Analysis:**
- PCA (0.99 variance) reduces 590 → ~150 features
- SMOTE/ADASYN balances training data
- StandardScaler normalizes features

**Conclusion:** Preprocessing pipeline impact > model architecture differences for this dataset.

---

### **3. Computational Efficiency**

| Model | Optimization Time | Speedup vs Baseline |
|-------|------------------|---------------------|
| LightGBM | 1.3 min | 49× faster |
| SVM | 1.5 min | 42× faster |
| XGBoost | 34. 1 min | 1.9× faster |
| Neural Net | 4.2 min | 15× faster |
| Random Forest | 63.4 min | Baseline |

*(10 population, 5 generations, SECOM)*

**Finding:** LightGBM and SVM best for rapid prototyping. 

---

### **4. GA Convergence Behavior**

- **Without adaptive mutation:** Converges in 2-5 generations (premature)
- **With adaptive mutation:** Maintains diversity, converges in 8-12 generations
- **Diversity threshold:** Setting to 0.0 (rely on patience only) performs best

---

## 🏆 Performance Benchmarks

### **SECOM Dataset (Full Run:  20 population, 15 generations)**

| Model | CV Score | Test Acc | Test F1 | Opt.  Time | Eval Time | Total Time |
|-------|----------|----------|---------|-----------|-----------|------------|
| Random Forest | 93.36% | 93.31% | 90.08% | 63.4 min | 2.1 s | 63.4 min |
| XGBoost | 93.36% | 93.31% | 90.08% | 34.1 min | 1.8 s | 34.1 min |
| LightGBM | 93.36% | 93.31% | 90.08% | 1.3 min | 1.2 s | 1.3 min |
| Neural Network | 93.36% | 93.31% | 90.08% | 4.2 min | 5.3 s | 4.2 min |
| SVM | 93.36% | 93.31% | 90.08% | 1.5 min | 3.1 s | 1.5 min |

**Hardware:** CPU-based (no GPU)

---

## Installation

### **Requirements**

- Python 3.9+
- 8GB+ RAM (for large datasets)
- Kaggle account (for dataset downloads)

### **Dependencies**

```bash
pip install -r requirements.txt
```

**Core:**
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0

**ML Models:**
- xgboost >= 1.5.0
- lightgbm >= 3.3.0

**Data:**
- pandas >= 1.3.0
- imbalanced-learn >= 0.9.0
- kaggle >= 1.5.0

**Visualization:**
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

**Documentation:**

- **[Configuration Guide](docs/CONFIGURATION_GUIDE.md)** - Customize GA parameters, mutation, selection
- **[Custom Datasets Guide](docs/CUSTOM_DATASETS.md)** - Add your own datasets
- **[Kaggle Setup](docs/KAGGLE_SETUP.md)** - Configure Kaggle API for dataset downloads

---

## Contributing

This is a research/thesis project. Contributions, suggestions, and bug reports are welcome! 

### **How to Contribute:**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## License

I don't know what this suppose to mean actually.

---

## Acknowledgments

- **Scikit-learn** team for comprehensive ML library
- **XGBoost** and **LightGBM** developers for efficient gradient boosting
- **Kaggle** for dataset hosting and API
- **SECOM**, **Fashion-MNIST**, **Isolet**, **Steel Plates** dataset creators
- Genetic Algorithm research community

---

## Contact

**Author:** HTHao  
**Project:** Genetic Algorithm Hyperparameter Optimization  
**Year:** 2026  

For questions, issues, or collaboration: 
- Open an issue on GitHub
- Email:  [theoneandonlyhth@gmail.com]

---

## Roadmap

### **Completed **
- [x] Core GA engine with 8 operators
- [x] 5 ML model optimizers
- [x] 4 benchmark datasets
- [x] Multi-objective optimization (NSGA-II)
- [x] Adaptive parameters
- [x] Statistical analysis tools
- [x] HTML report generation
- [x] Fitness metric comparison study

### **Potential Future Work**
- [ ] GPU acceleration for neural networks
- [ ] Distributed GA (parallel population evaluation)
- [ ] More datasets (UCI repository integration)
- [ ] Hyperparameter importance analysis
- [ ] Interactive web dashboard
- [ ] Docker containerization

---

**Feel free to use this repository if you find it useful! I would look forward to feedback as I am still very inexperience in programming field.**

**Happy Optimizing!**
