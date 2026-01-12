# Configuration Guide

Complete guide to customizing the Genetic Algorithm and model optimization parameters.

---

## Table of Contents

- [GA Configuration](#ga-configuration)
- [Model-Specific Parameters](#model-specific-parameters)
- [Preprocessing Configuration](#preprocessing-configuration)
- [Advanced GA Features](#advanced-ga-features)
- [Example Configurations](#example-configurations)

---

## GA Configuration

### **Basic Configuration**

```python
from ga. genetic_algorithm import GAConfig

config = GAConfig(
    population_size=20,      # Number of individuals in population
    num_generations=15,      # Maximum number of generations
    crossover_rate=0.8,      # Probability of crossover (0.0-1.0)
    mutation_rate=0.2,       # Probability of mutation (0.0-1.0)
    elitism_rate=0.1,        # Top % to preserve (0.0-1.0)
    random_state=42          # Seed for reproducibility
)
```

---

### **Population Parameters**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `population_size` | int | 50 | 10-100 | Number of individuals per generation |
| `num_generations` | int | 100 | 5-500 | Maximum generations to evolve |

**Guidelines:**
- **Small datasets (< 1000 samples):** 10-20 population
- **Medium datasets (1000-10000):** 20-50 population
- **Large datasets (> 10000):** 50-100 population

**Trade-off:** Larger population = better exploration, longer runtime

---

### **Operator Rates**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `crossover_rate` | float | 0.8 | 0.6-0.95 | Probability of combining two parents |
| `mutation_rate` | float | 0.1 | 0.05-0.3 | Probability of random changes |
| `elitism_rate` | float | 0.1 | 0.05-0.2 | Top % preserved unchanged |

**Guidelines:**
- **High exploration:** crossover=0.7, mutation=0.3
- **Balanced:** crossover=0.8, mutation=0.2 (recommended)
- **Exploitation:** crossover=0.9, mutation=0.1

---

### **Early Stopping**

```python
config = GAConfig(
    early_stopping=True,      # Enable early stopping
    patience=5,               # Generations without improvement
    diversity_threshold=0.05  # Min diversity (0.0 = disabled)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `early_stopping` | bool | True | Stop if no improvement |
| `patience` | int | 10 | Generations to wait before stopping |
| `diversity_threshold` | float | 0.0 | Min diversity to continue (0.0 = disabled) |

**When to use:**
- **Enable early stopping:** For quick experiments, limited time
- **Disable early stopping:** For thorough exploration, research

---

## Advanced GA Features

### **1. Adaptive Mutation**

Automatically adjusts mutation rate based on diversity or fitness stagnation. 

```python
config = GAConfig(
    # Enable adaptive mutation
    adaptive_mutation=True,
    mutation_method='adaptive',      # 'uniform', 'gaussian', 'polynomial', 'adaptive'
    mutation_strength='medium',      # 'small', 'medium', 'large'
    adaptive_method='diversity_based'  # 'diversity_based', 'fitness_based', 'schedule'
)
```

**Mutation Methods:**

| Method | Description | Best For |
|--------|-------------|----------|
| `uniform` | Random value from range | Discrete parameters |
| `gaussian` | Normal distribution around current | Fine-tuning |
| `polynomial` | Polynomial distribution | Continuous parameters |
| `adaptive` | Changes based on population state | General use |

**Mutation Strength:**

| Strength | Effect | Use Case |
|----------|--------|----------|
| `small` | Minor tweaks (±10%) | Fine-tuning, late generations |
| `medium` | Moderate changes (±30%) | Balanced exploration |
| `large` | Drastic changes (±60%) | Escaping local optima |

**Adaptive Methods:**

```python
# Diversity-based (recommended)
adaptive_method='diversity_based'  # Increases mutation when diversity drops

# Fitness-based
adaptive_method='fitness_based'    # Increases mutation when fitness stagnates

# Schedule-based
adaptive_method='schedule'         # Decreases mutation over time (simulated annealing)
```

---

### **2. Selection Methods**

```python
from ga.selection import SelectionMethod

config = GAConfig(
    selection_method=SelectionMethod.TOURNAMENT,  # Default
    tournament_size=3                             # For tournament selection
)
```

**Available Methods:**

| Method | Description | Diversity | Speed |
|--------|-------------|-----------|-------|
| `TOURNAMENT` | k individuals compete | Medium | Fast |
| `ROULETTE` | Fitness-proportional | Low | Medium |
| `RANK` | Rank-based selection | Medium | Medium |
| `SUS` | Stochastic Universal Sampling | High | Fast |
| `BOLTZMANN` | Temperature-based | Adaptive | Slow |

---

### **3. Crossover Methods**

```python
from ga.crossover import CrossoverMethod

config = GAConfig(
    crossover_method=CrossoverMethod. UNIFORM  # Default
)
```

**Available Methods:**

| Method | Description | Best For |
|--------|-------------|----------|
| `SINGLE_POINT` | Split at one point | Discrete genes |
| `TWO_POINT` | Split at two points | Preserving building blocks |
| `UNIFORM` | Random per gene | Mixed parameter types |
| `ARITHMETIC` | Weighted average | Continuous parameters |
| `BLX_ALPHA` | Blend crossover | Exploration |
| `SBX` | Simulated binary | Numerical optimization |

---

### **4. Diversity Maintenance**

Prevents premature convergence by maintaining population variety.

```python
config = GAConfig(
    diversity_maintenance=True,
    diversity_method='fitness_sharing',  # 'fitness_sharing', 'crowding'
    sharing_radius=0.1
)
```

**Methods:**

- **Fitness Sharing:** Penalizes similar individuals
- **Crowding:** Replaces similar parents with offspring
- **Niching:** Maintains sub-populations

---

## Model-Specific Parameters

### **Random Forest**

```python
from ga.ml_optimizer import MLOptimizer

# Default hyperparameter space
RF_TEMPLATE = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', None]
}

# Customize
optimizer = MLOptimizer(X, y, model_type='random_forest')
optimizer.chromosome_template['n_estimators'] = [100, 500, 1000]  # Larger forests
```

---

### **XGBoost**

```python
from models.xgboost_optimizer import XGBoostOptimizer, XGBOOST_HYPERPARAMETER_TEMPLATE

# Modify template
custom_template = XGBOOST_HYPERPARAMETER_TEMPLATE.copy()
custom_template['learning_rate'] = (0.001, 0.1)  # Slower learning
custom_template['max_depth'] = [3, 5, 7]         # Shallower trees

optimizer = XGBoostOptimizer(X, y)
optimizer.chromosome_template = custom_template
```

---

### **Neural Networks**

```python
from models.neural_network_optimizer import NeuralNetworkOptimizer

# Custom architecture search space
NN_TEMPLATE = {
    'hidden_layer_1':  [64, 128, 256, 512],   # Larger layers
    'hidden_layer_2': [0, 32, 64, 128],
    'hidden_layer_3':  [0, 16, 32],
    'learning_rate_init': (0.0001, 0.01),
    'solver': ['adam', 'sgd', 'lbfgs'],      # Add L-BFGS
    'activation': ['relu', 'tanh']
}

optimizer = NeuralNetworkOptimizer(X, y)
optimizer.chromosome_template = NN_TEMPLATE
```

---

## Preprocessing Configuration

### **Feature Scaling**

```python
# In hyperparameter template
'scaler': ['standard', 'minmax', 'robust']
```

| Scaler | Description | Best For |
|--------|-------------|----------|
| `standard` | Zero mean, unit variance | Most cases |
| `minmax` | Scale to [0, 1] | Neural networks, images |
| `robust` | Median/IQR-based | Outliers present |

---

### **Class Balancing**

```python
'smote_strategy': ['none', 'smote', 'adasyn', 'random_over', 'random_under']
```

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `none` | No balancing | Balanced datasets |
| `smote` | Synthetic samples | Moderate imbalance |
| `adasyn` | Adaptive synthetic | Severe imbalance |
| `random_over` | Duplicate minority | Simple approach |
| `random_under` | Remove majority | Large datasets |

---

### **Dimensionality Reduction**

```python
'pca_variance':  [0.90, 0.95, 0.99]
```

| Variance | Components Kept | Trade-off |
|----------|----------------|-----------|
| 0.90 | ~50-70% | Fast, less info |
| 0.95 | ~70-85% | Balanced |
| 0.99 | ~90-98% | Slow, more info |

---

## Example Configurations

### **1. Quick Prototyping (Fast)**

```python
config = GAConfig(
    population_size=10,
    num_generations=5,
    early_stopping=True,
    patience=3,
    verbose=1
)
```

**Runtime:** 2-5 minutes  
**Use:** Testing, debugging, initial exploration

---

### **2. Standard Research (Balanced)**

```python
config = GAConfig(
    population_size=20,
    num_generations=15,
    crossover_rate=0.8,
    mutation_rate=0.2,
    elitism_rate=0.1,
    
    adaptive_mutation=True,
    mutation_method='adaptive',
    mutation_strength='medium',
    
    early_stopping=True,
    patience=5,
    verbose=1
)
```

**Runtime:** 30-60 minutes  
**Use:** Thesis work, publications

---

### **3. Thorough Exploration (Slow)**

```python
config = GAConfig(
    population_size=50,
    num_generations=30,
    crossover_rate=0.8,
    mutation_rate=0.25,
    elitism_rate=0.15,
    
    adaptive_mutation=True,
    mutation_strength='large',
    diversity_maintenance=True,
    
    early_stopping=False,  # Run all generations
    verbose=2
)
```

**Runtime:** 2-4 hours  
**Use:** Final experiments, benchmarking

---

### **4. Imbalanced Data (SECOM-style)**

```python
config = GAConfig(
    population_size=20,
    num_generations=15,
    
    # Force exploration of balancing methods
    adaptive_mutation=True,
    mutation_strength='large'
)

# Custom fitness:  Macro F1
def macro_f1_fitness(config):
    # ...  training code ...
    from sklearn.metrics import f1_score
    return f1_score(y_val, y_pred, average='macro')

optimizer. fitness_function = macro_f1_fitness
```

**Key:** Use macro-averaged metrics, try class_weight='balanced'

---

### **5. Multi-Objective Optimization**

```python
from ga.multi_objective import NSGAII

# Optimize accuracy AND speed
def fitness_multi(config):
    accuracy = ...   # Train and evaluate
    speed = 1. 0 / training_time
    return [accuracy, speed]  # Return list of objectives

nsga2 = NSGAII(
    population_size=30,
    num_generations=20,
    objectives=['maximize', 'maximize']
)

pareto_front = nsga2.optimize(fitness_multi, chromosome_template)
```

---

## Runtime Configuration

### **Logging Verbosity**

```python
config = GAConfig(
    verbose=0  # Silent
    verbose=1  # Normal (generations, best fitness)
    verbose=2  # Detailed (per-individual evaluation)
)
```

---

### **Fitness Caching**

```python
config = GAConfig(
    cache_fitness=True  # Cache evaluations (faster, more memory)
    cache_fitness=False  # No caching (slower, less memory)
)
```

**When to use:**
- **Enable:** Deterministic fitness, limited hyperparameter space
- **Disable:** Stochastic models, large search space

---

## Hyperparameter Tuning Tips

### **1. Start Small**
```python
# Initial run:  10 population, 5 generations
# Refine: 20 population, 15 generations
# Final: 50 population, 30 generations
```

### **2. Monitor Diversity**
- If diversity drops to 0.0 early → increase mutation_rate
- If diversity stays high → working well! 

### **3. Adaptive Mutation Recommended**
```python
adaptive_mutation=True
mutation_strength='large'  # For imbalanced data
```

### **4. Use Appropriate Metrics**
- **Balanced data:** Accuracy
- **Imbalanced data:** Macro F1, Balanced Accuracy

---

## Configuration Impact

| Change | Effect on Runtime | Effect on Quality |
|--------|------------------|-------------------|
| +10 population | +50% | +10-15% |
| +10 generations | +50% | +5-10% |
| Enable adaptive | +5% | +15-20% |
| Disable early stopping | +100-200% | +2-5% |

---

## Troubleshooting

### **Converges too early (Gen 1-3)**
```python
# Increase mutation
mutation_rate=0.3
adaptive_mutation=True
mutation_strength='large'
```

### **Never converges (still improving at Gen 30)**
```python
# Increase generations or enable early stopping
num_generations=50
patience=10
```

### **All individuals identical**
```python
# Expand hyperparameter space
# Or increase mutation
diversity_threshold=0.0  # Don't stop on low diversity
```

---

**For more examples, see the scripts:**
- `compare_models.py` - Standard configuration
- `compare_fitness_metrics.py` - Adaptive mutation example

---

**Happy Tuning!**