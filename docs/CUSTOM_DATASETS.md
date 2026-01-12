# Custom Dataset Guide

Complete guide to adding your own datasets to the optimization framework.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Dataset Format Requirements](#dataset-format-requirements)
- [Method 1: CSV Files](#method-1-csv-files)
- [Method 2: NumPy Arrays](#method-2-numpy-arrays)
- [Method 3: Register in Dataset Loader](#method-3-register-in-dataset-loader)
- [Example Workflows](#example-workflows)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### **Option A: Direct Loading (Simplest)**

```python
import numpy as np
import pandas as pd
from ga.ml_optimizer import MLOptimizer

# Load your CSV
df = pd.read_csv('your_dataset.csv')
X = df.drop('target', axis=1).values
y = df['target'].values

# Optimize! 
optimizer = MLOptimizer(X, y, model_type='random_forest')
results = optimizer.optimize()
```

---

### **Option B: Register for Reuse**

```python
# Add to preprocessing/data_loader.py registry
# Then use like built-in datasets
loader = DatasetLoader()
X, y, metadata = loader.load_dataset('your_dataset')
```

---

## Dataset Format Requirements

### **Feature Matrix (X)**
- **Type:** `numpy.ndarray` or `pandas.DataFrame`
- **Shape:** `(n_samples, n_features)`
- **Data type:** Numerical (int, float)
- **Missing values:** Allowed (will be imputed)

### **Labels (y)**
- **Type:** `numpy.ndarray` or `pandas.Series`
- **Shape:** `(n_samples,)`
- **Data type:** 
  - Classification: int (0, 1, 2, ...)
  - Binary: 0/1 or -1/1

### **Example:**
```python
# Valid dataset
X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])  # (2 samples, 3 features)
y = np.array([0, 1])              # (2 samples,)
```

---

## Method 1: CSV Files

### **Step 1: Prepare CSV**

**Format:**
```csv
feature1,feature2,feature3,target
1.0,2.0,3.0,0
4.0,5.0,6.0,1
7.0,8.0,9.0,0
```

**Requirements:**
- Header row with column names
- Target column named `target`, `label`, `class`, or `y`
- No index column (or will be skipped)
- Numerical features only

---

### **Step 2: Load and Use**

```python
import pandas as pd
from ga.ml_optimizer import MLOptimizer

# Load CSV
df = pd.read_csv('datasets/custom/my_data.csv')

# Separate features and target
target_col = 'target'  # Or 'label', 'class', etc.
X = df.drop(target_col, axis=1).values
y = df[target_col].values

# Optimize
optimizer = MLOptimizer(X, y, model_type='xgboost')
results = optimizer.optimize()
```

---

### **Step 3: Handle Categorical Features** (if any)

```python
from sklearn.preprocessing import LabelEncoder

# Encode categorical features
df['category'] = LabelEncoder().fit_transform(df['category'])

# Or one-hot encoding
df = pd.get_dummies(df, columns=['category'])
```

---

## Method 2: NumPy Arrays

### **From Existing Arrays**

```python
import numpy as np
from models.lightgbm_optimizer import LightGBMOptimizer

# Your data (from any source)
X = np.random.rand(1000, 50)  # 1000 samples, 50 features
y = np. random.randint(0, 2, 1000)  # Binary classification

# Optimize
optimizer = LightGBMOptimizer(X, y)
results = optimizer.optimize()
```

---

### **From scikit-learn Datasets**

```python
from sklearn.datasets import make_classification, load_breast_cancer
from ga.ml_optimizer import MLOptimizer

# Synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Or real dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Optimize
optimizer = MLOptimizer(X, y, model_type='svm')
results = optimizer.optimize()
```

---

## Method 3: Register in Dataset Loader

For datasets you'll use repeatedly, register them in the data loader. 

### **Step 1: Add to Registry**

```cmd
notepad config\datasets_registry.json
```

**Add your dataset:**

```json
{
  "your_dataset": {
    "name": "Your Dataset Name",
    "source": "local",
    "path": "datasets/custom/your_data.csv",
    "description": "Brief description",
    "features": 20,
    "samples": 1000,
    "classes": 2,
    "task": "classification"
  }
}
```

---

### **Step 2: Create Parser (if needed)**

If CSV format is non-standard, create a custom parser:

```cmd
notepad preprocessing\file_parsers\your_parser.py
```

```python
import pandas as pd
import numpy as np

def parse_your_dataset(filepath):
    """
    Parse your custom dataset format. 
    
    Returns:
        X (np.ndarray): Features
        y (np.ndarray): Labels
    """
    # Custom parsing logic
    df = pd.read_csv(filepath, sep=';')  # Semicolon-separated
    
    # Extract features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return X, y
```

---

### **Step 3: Register Parser**

```cmd
notepad preprocessing\data_loader.py
```

**Find `_parse_files` method and add:**

```python
elif dataset_name == 'your_dataset': 
    from preprocessing.file_parsers.your_parser import parse_your_dataset
    X, y = parse_your_dataset(dataset_path / 'your_data.csv')
```

---

### **Step 4: Use Like Built-in Datasets**

```python
from preprocessing.data_loader import DatasetLoader

loader = DatasetLoader()
X, y, metadata = loader.load_dataset('your_dataset')

# Now use in any optimizer
from ga.ml_optimizer import MLOptimizer
optimizer = MLOptimizer(X, y)
results = optimizer.optimize()
```

---

## Example Workflows

### **Example 1: Kaggle Competition Dataset**

```bash
# Download from Kaggle
kaggle competitions download -c <competition-name>

# Extract
unzip <competition-name>.zip -d datasets/custom/
```

```python
import pandas as pd
from ga.ml_optimizer import MLOptimizer

# Load
df = pd.read_csv('datasets/custom/train.csv')

# Preprocess
X = df.drop(['id', 'target'], axis=1).values
y = df['target'].values

# Optimize
optimizer = MLOptimizer(X, y, model_type='xgboost')
results = optimizer.optimize()
```

---

### **Example 2: UCI ML Repository**

```python
from sklearn.datasets import fetch_openml

# Fetch dataset
data = fetch_openml('diabetes', version=1, as_frame=False)
X, y = data.data, data.target

# Convert categorical labels to numeric
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

# Optimize
from models.neural_network_optimizer import NeuralNetworkOptimizer
optimizer = NeuralNetworkOptimizer(X, y)
results = optimizer.optimize()
```

---

### **Example 3: Image Data (Flattened)**

```python
from tensorflow.keras.datasets import mnist
import numpy as np

# Load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten images:  28x28 → 784
X_train = X_train. reshape(-1, 784) / 255.0  # Normalize to [0,1]
X_test = X_test.reshape(-1, 784) / 255.0

# Combine train+test for GA optimization
X = np.vstack([X_train, X_test])
y = np.hstack([y_train, y_test])

# Optimize
from models.lightgbm_optimizer import LightGBMOptimizer
optimizer = LightGBMOptimizer(X, y)
results = optimizer.optimize()
```

---

### **Example 4: Time Series (Windowed)**

```python
import pandas as pd
import numpy as np

# Load time series
df = pd.read_csv('stock_prices.csv')

# Create sliding windows
window_size = 10
X, y = [], []

for i in range(len(df) - window_size):
    X.append(df.iloc[i:i+window_size]['price'].values)
    y.append(1 if df.iloc[i+window_size]['price'] > df.iloc[i+window_size-1]['price'] else 0)

X = np.array(X)
y = np.array(y)

# Optimize
from ga.ml_optimizer import MLOptimizer
optimizer = MLOptimizer(X, y, model_type='random_forest')
results = optimizer.optimize()
```

---

## 🛠️ Preprocessing Custom Datasets

### **1. Handle Missing Values**

```python
import numpy as np

# Check for missing values
if np.isnan(X).any():
    print(f"Missing values:  {np.isnan(X).sum()}")
    
    # Option A: Let optimizer handle it (uses mean imputation)
    # No action needed - preprocessing pipeline handles it
    
    # Option B:  Manual imputation
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
```

---

### **2. Encode Categorical Labels**

```python
from sklearn.preprocessing import LabelEncoder

# If labels are strings:  ['cat', 'dog'] → [0, 1]
if y.dtype == 'object': 
    le = LabelEncoder()
    y = le.fit_transform(y)
```

---

### **3. Feature Engineering**

```python
import pandas as pd

df = pd.read_csv('your_data.csv')

# Create new features
df['feature_ratio'] = df['feature1'] / (df['feature2'] + 1e-8)
df['feature_product'] = df['feature1'] * df['feature3']

# Log transform skewed features
df['feature1_log'] = np.log1p(df['feature1'])

# Extract X, y
X = df.drop('target', axis=1).values
y = df['target'].values
```

---

## Common Issues & Solutions

### **Issue 1: "ValueError: could not convert string to float"**

**Cause:** Categorical features not encoded

**Solution:**
```python
# Encode categorical columns
from sklearn.preprocessing import LabelEncoder

for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])
```

---

### **Issue 2: "Shape mismatch:  X and y different lengths"**

**Cause:** Dropped rows without updating y

**Solution:**
```python
# Drop rows from both X and y
mask = df['feature1'].notna()  # Example: keep non-null
X = df[mask].drop('target', axis=1).values
y = df[mask]['target'].values
```

---

### **Issue 3: "Memory Error"**

**Cause:** Dataset too large

**Solution:**
```python
# Sample subset
from sklearn.model_selection import train_test_split
X_subset, _, y_subset, _ = train_test_split(X, y, train_size=0.1, stratify=y)

# Or use smaller GA population
config = GAConfig(population_size=10, num_generations=5)
```

---

### **Issue 4: "All individuals have fitness 0.0"**

**Cause:** Invalid data (NaN, Inf) or wrong target format

**Solution:**
```python
# Check for invalid values
print(f"NaN in X: {np.isnan(X).any()}")
print(f"Inf in X: {np.isinf(X).any()}")
print(f"Unique labels: {np.unique(y)}")

# Clean
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Ensure binary/multiclass labels start at 0
y = y - y.min()
```

---

## Dataset Statistics Helper

Create a utility to check your dataset: 

```python
import numpy as np
import pandas as pd

def check_dataset(X, y):
    """Print dataset statistics."""
    print("="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Missing values: {np. isnan(X).sum()} ({np.isnan(X).sum()/X.size*100:.2f}%)")
    print(f"Feature dtype: {X.dtype}")
    print(f"Label dtype: {y.dtype}")
    print(f"Memory usage: {X.nbytes / 1024**2:.2f} MB")
    print("="*60)

# Use it
check_dataset(X, y)
```

---

## Checklist Before Optimization

- [ ] X is 2D numpy array `(n_samples, n_features)`
- [ ] y is 1D numpy array `(n_samples,)`
- [ ] No string values in X
- [ ] Labels are integers starting from 0
- [ ] No infinite values (check with `np.isinf()`)
- [ ] Reasonable size (< 100K samples for quick tests)

---

## Additional Resources

- **Scikit-learn datasets:** https://scikit-learn.org/stable/datasets. html
- **Kaggle datasets:** https://www.kaggle.com/datasets
- **UCI ML Repository:** https://archive.ics.uci.edu/ml/index.php
- **OpenML:** https://www.openml.org/

---

**For more examples, see:**
- `preprocessing/data_loader.py` - Built-in dataset loading
- `preprocessing/file_parsers/` - Custom parsers
- `compare_models.py` - Example usage

---

**Happy Dataset Wrangling! **