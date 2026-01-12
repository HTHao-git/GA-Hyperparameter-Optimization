# Test loading configs for all 3 model types
from utils.config_loader import get_full_config
from utils.colors import print_header, print_success, print_key_value

print_header("TESTING ALL MODEL CONFIGURATIONS")

# Test 1: Neural Network (default)
print("\n" + "="*70)
print("TEST 1: Neural Network Configuration")
print("="*70)

config_nn = get_full_config()
print_success(f"Loaded Neural Network config")
print_key_value("  Model Type", config_nn['model']['type'])
print_key_value("  Dataset", config_nn['dataset']['name'])
print_key_value("  HPO Method", config_nn['hpo']['method'])
print_key_value("  Num Hyperparameters", len(config_nn['hyperparameters']))

# Test 2: SVM (modify config)
print("\n" + "="*70)
print("TEST 2: SVM Configuration")
print("="*70)

# Temporarily modify default_config.yaml or create custom config
# For now, let's just load the SVM hyperparameter config directly
from utils.config_loader import load_hyperparameter_config, apply_dataset_overrides

svm_config = load_hyperparameter_config('svm')
svm_config = apply_dataset_overrides(svm_config, 'secom')
print_success(f"Loaded SVM config")
print_key_value("  Model Type", svm_config['model_type'])
print_key_value("  Num Hyperparameters", len(svm_config['hyperparameters']))

# Test 3: Logistic Regression
print("\n" + "="*70)
print("TEST 3: Logistic Regression Configuration")
print("="*70)

lr_config = load_hyperparameter_config('logistic_regression')
lr_config = apply_dataset_overrides(lr_config, 'steel_plates')
print_success(f"Loaded Logistic Regression config")
print_key_value("  Model Type", lr_config['model_type'])
print_key_value("  Num Hyperparameters", len(lr_config['hyperparameters']))

print("\n" + "="*70)
print_success("All config tests passed!")
print("="*70)