# ============================================================================
# CONFIG LOADER - Load and validate YAML/JSON configuration files
# ============================================================================
# This module handles all configuration loading with validation and defaults.
# 
# USAGE: 
#   from utils.config_loader import load_config
#   config = load_config('config/default_config.yaml')
# 
# Last updated: 2025-12-31
# ============================================================================

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / 'config'
HYPERPARAM_DIR = CONFIG_DIR / 'hyperparameters'


# ============================================================================
# LOAD YAML CONFIG
# ============================================================================

def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is invalid YAML
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {filepath}: {e}")


# ============================================================================
# LOAD JSON CONFIG
# ============================================================================

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON configuration file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary containing configuration
        
    Raises: 
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is invalid JSON
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")


# ============================================================================
# LOAD MASTER CONFIG
# ============================================================================

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load master configuration file (YAML).
    
    Args:
        config_path: Path to config file (default: config/default_config.yaml)
        
    Returns:
        Dictionary containing full configuration
    """
    if config_path is None:
        config_path = CONFIG_DIR / 'default_config.yaml'
    
    config = load_yaml(config_path)
    
    print(f"Loaded master config:  {config_path}")
    
    return config


# ============================================================================
# LOAD HYPERPARAMETER CONFIG
# ============================================================================

def load_hyperparameter_config(model_type: str, 
                               custom_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load hyperparameter configuration for a specific model type.
    
    Args:
        model_type: Model type ('neural_network', 'svm', 'logistic_regression')
        custom_path: Custom config path (optional)
        
    Returns:
        Dictionary containing hyperparameter configuration
        
    Raises:
        ValueError: If model_type is invalid
        FileNotFoundError: If config file doesn't exist
    """
    valid_models = ['neural_network', 'svm', 'logistic_regression']
    
    if model_type not in valid_models:
        raise ValueError(
            f"Invalid model_type '{model_type}'."
            f"Must be one of: {valid_models}"
        )
    
    # Use custom path or default
    if custom_path: 
        config_path = Path(custom_path)
    else:
        config_path = HYPERPARAM_DIR / f'{model_type}.json'
    
    config = load_json(config_path)
    
    print(f"Loaded hyperparameter config:  {config_path}")
    
    return config


# ============================================================================
# APPLY DATASET-SPECIFIC OVERRIDES
# ============================================================================

def apply_dataset_overrides(hyperparam_config: Dict[str, Any], 
                            dataset_name: str) -> Dict[str, Any]:
    """
    Apply dataset-specific hyperparameter overrides.
    
    For example, SECOM uses different PCA ranges than ISOLET.
    
    Args:
        hyperparam_config: Base hyperparameter configuration
        dataset_name: Dataset name (e.g., 'secom', 'isolet')
        
    Returns:
        Updated hyperparameter configuration
    """
    overrides = hyperparam_config.get('dataset_specific_overrides', {})
    
    if dataset_name in overrides:
        dataset_overrides = overrides[dataset_name]
        
        for param_name, param_override in dataset_overrides.items():
            if param_name == 'notes':
                continue  # Skip notes field
            
            if param_name in hyperparam_config['hyperparameters']:
                # Merge override into base config
                hyperparam_config['hyperparameters'][param_name].update(param_override)
        
        print(f"Applied dataset-specific overrides for '{dataset_name}'")
    
    return hyperparam_config


# ============================================================================
# VALIDATE CONFIG
# ============================================================================

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate master configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required top-level keys
    required_keys = ['experiment', 'dataset', 'model', 'hpo', 'output']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: '{key}'")
    
    # Validate dataset name
    valid_datasets = ['isolet', 'fashion_mnist', 'secom', 'steel_plates', 'custom']
    dataset_name = config['dataset']['name']
    
    if dataset_name not in valid_datasets: 
        raise ValueError(
            f"Invalid dataset '{dataset_name}'."
            f"Must be one of: {valid_datasets}"
        )
    
    # Validate model type
    valid_models = ['neural_network', 'svm', 'logistic_regression']
    model_type = config['model']['type']
    
    if model_type not in valid_models: 
        raise ValueError(
            f"Invalid model type '{model_type}'."
            f"Must be one of:  {valid_models}"
        )
    
    # Validate HPO method
    valid_hpo = ['nsga2', 'vanilla_ga', 'grid_search']
    hpo_method = config['hpo']['method']
    
    if hpo_method not in valid_hpo: 
        raise ValueError(
            f"Invalid HPO method '{hpo_method}'."
            f"Must be one of: {valid_hpo}"
        )
    
    # Validate split ratios (if not using SMOTE)
    split_ratio = config['dataset']['split_ratio']
    total = split_ratio['train'] + split_ratio['validation'] + split_ratio['test']
    
    if abs(total - 1.0) > 0.01:  # Allow small floating point error
        raise ValueError(
            f"Split ratios must sum to 1.0 (got {total})."
            f"Current:  train={split_ratio['train']}, "
            f"val={split_ratio['validation']}, "
            f"test={split_ratio['test']}"
        )
    
    print("Configuration validated successfully")
    
    return True


# ============================================================================
# GET FULL CONFIG (Master + Hyperparameters)
# ============================================================================

def get_full_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and merge master config + hyperparameter config.
    
    This is the main function you'll use in your code.
    
    Args:
        config_path: Path to master config (default: config/default_config.yaml)
        
    Returns: 
        Complete configuration dictionary with all settings
    """
    # Load master config
    config = load_config(config_path)
    
    # Validate master config
    validate_config(config)
    
    # Load hyperparameter config for chosen model
    model_type = config['model']['type']
    custom_hyperparam_path = config['model'].get('hyperparameter_config')
    
    hyperparam_config = load_hyperparameter_config(
        model_type, 
        custom_path=custom_hyperparam_path
    )
    
    # Apply dataset-specific overrides
    dataset_name = config['dataset']['name']
    hyperparam_config = apply_dataset_overrides(hyperparam_config, dataset_name)
    
    # Merge into main config
    config['hyperparameters'] = hyperparam_config['hyperparameters']
    config['hyperparameter_metadata'] = {
        'model_type': hyperparam_config['model_type'],
        'description': hyperparam_config['description'],
        'last_updated':  hyperparam_config['last_updated']
    }
    
    print("Full configuration loaded and merged")
    
    return config


# ============================================================================
# DISPLAY CONFIG (Pretty Print)
# ============================================================================

def display_config(config: Dict[str, Any]):
    """
    Pretty-print configuration for review.
    
    Args:
        config: Configuration dictionary
    """
    from tabulate import tabulate
    
    print("\n" + "="*70)
    print("📋 EXPERIMENT CONFIGURATION")
    print("="*70)
    
    # Experiment info
    exp = config['experiment']
    print(f"\n🔬 Experiment:")
    print(f"   Name: {exp['name'] if exp['name'] else 'Auto-generated'}")
    print(f"   Random Seed: {exp['random_seed']}")
    print(f"   Description: {exp['description']}")
    
    # Dataset info
    ds = config['dataset']
    print(f"\n📊 Dataset:")
    print(f"   Name: {ds['name']}")
    print(f"   Split: {ds['split_ratio']['train']:.0%} train / {ds['split_ratio']['validation']:.0%} val / {ds['split_ratio']['test']:.0%} test")
    print(f"   SMOTE: {ds['preprocessing']['use_smote']}")
    print(f"   Standardize: {ds['preprocessing']['standardize']}")
    
    # Model info
    model = config['model']
    print(f"\n🤖 Model:")
    print(f"   Type: {model['type']}")
    print(f"   Training Epochs: {model['training']['epochs']}")
    print(f"   Final Epochs: {model['training']['final_epochs']}")
    
    # HPO info
    hpo = config['hpo']
    print(f"\n🧬 HPO Method:")
    print(f"   Method: {hpo['method']}")
    print(f"   Population:  {hpo.get('population_size', 'default')}")
    print(f"   Generations: {hpo.get('generations', 'default')}")
    print(f"   Resume: {hpo['resume']}")
    
    # Output info
    out = config['output']
    print(f"\n💾 Output:")
    print(f"   Directory: {out['base_dir']}")
    print(f"   Checkpoint Every: {out['checkpoint_frequency']} generations")
    print(f"   Generate Plots: {out['generate_plots']}")
    
    print("\n" + "="*70)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Test loading configuration
    try:
        config = get_full_config()
        display_config(config)
        
        print("\nConfig loader test passed!")
        
    except Exception as e: 
        print(f"\n❌ Error:  {e}")
        import traceback
        traceback.print_exc()