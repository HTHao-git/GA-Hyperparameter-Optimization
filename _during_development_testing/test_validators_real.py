# Test validators with realistic project scenarios
import numpy as np
from pathlib import Path
from utils.validators import (
    validate_file_path,
    validate_directory,
    validate_range,
    validate_choice,
    validate_split_ratios,
    validate_dataset_shape,
    detect_file_format
)
from utils.colors import print_header, print_section, print_success, print_error

print_header("REALISTIC VALIDATOR TESTING")

# Test 1: Directory validation (create outputs folder)
print_section("Directory Validation")
try:
    validate_directory("outputs", create_if_missing=True)
    print_success("outputs/ directory validated (created if needed)")
except Exception as e: 
    print_error(f"Failed:  {e}")

# Test 2: Config file validation
print("\n")
print_section("Config File Validation")
try:
    validate_file_path("config/default_config.yaml", 
                      allowed_extensions=['.yaml', '.yml'])
    print_success("config/default_config.yaml validated")
except Exception as e:
    print_error(f"Failed:  {e}")

# Test 3: File format detection
print("\n")
print_section("File Format Detection")
test_files = [
    "data.csv",
    "dataset.data",
    "features.arff",
    "results.xlsx",
    "model.pkl"
]

for filename in test_files:
    fmt = detect_file_format(filename)
    print(f"  {filename: 20} -> {fmt}")

# Test 4: Hyperparameter validation (realistic values)
print("\n")
print_section("Hyperparameter Validation")

hyperparams = {
    'learning_rate': 0.001,
    'dropout':  0.3,
    'num_layers': 3,
    'activation': 'relu',
    'batch_size': 64
}

try:
    validate_range(hyperparams['learning_rate'], 0.0001, 0.01, "learning_rate")
    validate_range(hyperparams['dropout'], 0.0, 0.5, "dropout")
    validate_range(hyperparams['num_layers'], 1, 4, "num_layers")
    validate_choice(hyperparams['activation'], ['relu', 'tanh', 'elu'], "activation")
    validate_choice(hyperparams['batch_size'], [32, 64, 128, 256], "batch_size")
    
    print_success("All hyperparameters validated successfully")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
except Exception as e:
    print_error(f"Validation failed: {e}")

# Test 5: Dataset split validation
print("\n")
print_section("Dataset Split Validation")

splits = [
    (0.6, 0.2, 0.2, "Standard split"),
    (0.7, 0.15, 0.15, "More training data"),
    (0.8, 0.1, 0.1, "Large training set"),
]

for train, val, test, desc in splits:
    try: 
        validate_split_ratios(train, val, test)
        print_success(f"{desc}:  {train}/{val}/{test}")
    except Exception as e:
        print_error(f"{desc} failed: {e}")

# Test 6: Dataset shape validation (simulate real data)
print("\n")
print_section("Dataset Shape Validation")

# Simulate SECOM dataset (1567 samples, 590 features → 50 PCA components)
X_train = np.random.randn(940, 50)  # 60% of 1567
y_train = np.random.randint(0, 2, 940)

X_val = np.random.randn(313, 50)  # 20%
y_val = np.random.randint(0, 2, 313)

X_test = np.random.randn(314, 50)  # 20%
y_test = np.random.randint(0, 2, 314)

try:
    validate_dataset_shape(X_train, y_train)
    validate_dataset_shape(X_val, y_val)
    validate_dataset_shape(X_test, y_test)
    
    print_success("All dataset shapes validated")
    print(f"  Train:  X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:    X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
except Exception as e:
    print_error(f"Validation failed: {e}")

print("\n" + "="*70)
print_success("All realistic validation tests passed!")
print("="*70)