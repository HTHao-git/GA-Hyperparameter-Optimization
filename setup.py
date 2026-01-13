# ============================================================================
# SETUP & VERIFICATION SCRIPT
# ============================================================================
# Automated setup for GA Hyperparameter Optimization Framework
#
# FEATURES:
#   - Dependency checking
#   - Dataset downloading
#   - Module verification
#   - System diagnostics
#
# USAGE:
#   python setup.py
#
# Last updated: 2026-01-13
# ============================================================================

import sys
import subprocess
import importlib
from pathlib import Path
import time

# Terminal colors (Windows-compatible)
try:
    import colorama
    colorama.init()
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
except ImportError:
    GREEN = RED = YELLOW = BLUE = RESET = BOLD = ''


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text):
    """Print section header."""
    print()
    print("=" * 80)
    print(f"{BOLD}{text}{RESET}")
    print("=" * 80)
    print()


def print_success(text):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text):
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_info(text):
    """Print info message."""
    print(f"{BLUE}ℹ {text}{RESET}")


def print_step(step, total, text):
    """Print step progress."""
    print(f"\n{BOLD}[{step}/{total}] {text}{RESET}")


# ============================================================================
# DEPENDENCY CHECKING
# ============================================================================

REQUIRED_PACKAGES = {
    'numpy': 'numpy>=1.21.0',
    'scipy': 'scipy>=1.7.0',
    'sklearn': 'scikit-learn>=1.0.0',
    'xgboost': 'xgboost>=1.5.0',
    'lightgbm': 'lightgbm>=3.3.0',
    'matplotlib': 'matplotlib>=3.4.0',
    'seaborn': 'seaborn>=0.11.0',
    'pandas': 'pandas>=1.3.0',
    'imblearn': 'imbalanced-learn>=0.9.0',
    'kaggle':  'kaggle>=1.5.0'
}


def check_python_version():
    """Check if Python version is adequate."""
    print_step(1, 6, "Checking Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"  Python version: {version_str}")
    
    if version.major >= 3 and version.minor >= 9:
        print_success(f"Python {version_str} is compatible")
        return True
    else:
        print_error(f"Python 3.9+ required (found {version_str})")
        return False


def check_dependencies():
    """Check if all required packages are installed."""
    print_step(2, 6, "Checking Dependencies")
    
    missing = []
    installed = []
    
    for package, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(package)
            installed.append(package)
            print_success(f"{package:15} installed")
        except ImportError: 
            missing.append(pip_name)
            print_error(f"{package:15} MISSING")
    
    print()
    
    if missing: 
        print_warning(f"Missing {len(missing)} package(s)")
        print()
        print("Install missing packages with:")
        print(f"  {BOLD}pip install {' '.join(missing)}{RESET}")
        print()
        
        response = input("Install missing packages now? [y/N]: ").strip().lower()
        
        if response == 'y': 
            print()
            print_info("Installing packages...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
                print_success("All packages installed!")
                return True
            except subprocess.CalledProcessError: 
                print_error("Installation failed. Please install manually.")
                return False
        else:
            return False
    else:
        print_success(f"All {len(installed)} required packages installed")
        return True


# ============================================================================
# KAGGLE API SETUP
# ============================================================================

def check_kaggle_setup():
    """Check if Kaggle API is configured."""
    print_step(3, 6, "Checking Kaggle API Configuration")
    
    # Check if kaggle module exists
    try:
        import kaggle
    except ImportError:
        print_error("Kaggle package not installed")
        return False
    
    # Check for kaggle.json
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print_success(f"Found kaggle.json at {kaggle_json}")
        
        # Try to authenticate
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print_success("Kaggle API authenticated successfully")
            return True
        except Exception as e:
            print_error(f"Kaggle authentication failed: {e}")
            return False
    else:
        print_warning("Kaggle API credentials not found")
        print()
        print("To set up Kaggle API:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New API Token'")
        print("  4. Save kaggle.json to:")
        print(f"     {kaggle_dir}")
        print()
        
        response = input("Have you set up kaggle.json? [y/N]: ").strip().lower()
        
        if response == 'y':
            return check_kaggle_setup()  # Retry
        else:
            print_info("You can set up Kaggle later (see docs/KAGGLE_SETUP.md)")
            return False


# ============================================================================
# DATASET DOWNLOADING
# ============================================================================

DATASETS = {
    'secom': {
        'name': 'SECOM (Semiconductor Manufacturing)',
        'source': 'UCI ML Repository via Kaggle',
        'kaggle_dataset': 'paresh2047/uci-semcom',
        'size': '~5 MB'
    },
    'fashion_mnist': {
        'name':  'Fashion-MNIST',
        'source': 'Zalando Research',
        'kaggle_dataset': 'zalando-research/fashionmnist',
        'size': '~30 MB'
    },
    'isolet': {
        'name':  'Isolet (Speech Recognition)',
        'source': 'UCI ML Repository',
        'kaggle_dataset':  'uciml/isolet',
        'size': '~10 MB'
    },
    'steel_plates': {
        'name': 'Steel Plates Faults',
        'source': 'UCI ML Repository',
        'kaggle_dataset': 'uciml/faulty-steel-plates',
        'size': '~2 MB'
    }
}


def check_datasets():
    """Check which datasets are available locally."""
    print_step(4, 6, "Checking Datasets")
    
    datasets_dir = Path('datasets')
    datasets_dir.mkdir(exist_ok=True)
    
    available = []
    missing = []
    
    for dataset_key, info in DATASETS.items():
        dataset_path = datasets_dir / dataset_key
        
        if dataset_path.exists() and any(dataset_path.iterdir()):
            available.append(dataset_key)
            print_success(f"{info['name']:40} available")
        else:
            missing.append(dataset_key)
            print_warning(f"{info['name']:40} MISSING")
    
    print()
    
    if missing: 
        print_info(f"{len(missing)} dataset(s) not downloaded")
        print()
        
        response = input("Download missing datasets now?  [y/N]: ").strip().lower()
        
        if response == 'y':
            return download_datasets(missing)
        else:
            print_info("You can download datasets later using the data loader")
            return len(available) > 0
    else: 
        print_success(f"All {len(available)} datasets available")
        return True


def download_datasets(dataset_keys):
    """Download specified datasets."""
    from preprocessing.data_loader import DatasetLoader
    from utils.logger import get_logger
    
    logger = get_logger(name="SETUP", verbose=True)
    
    print()
    print_info("Downloading datasets...")
    print()
    
    loader = DatasetLoader(logger=logger, interactive=False)
    
    success_count = 0
    
    for dataset_key in dataset_keys:
        try:
            print()
            print_info(f"Downloading {DATASETS[dataset_key]['name']}...")
            X, y, metadata = loader.load_dataset(dataset_key)
            print_success(f"Downloaded {dataset_key}:  {X.shape[0]} samples, {X.shape[1]} features")
            success_count += 1
        except Exception as e:
            print_error(f"Failed to download {dataset_key}: {e}")
    
    print()
    
    if success_count == len(dataset_keys):
        print_success(f"All {success_count} datasets downloaded successfully")
        return True
    else:
        print_warning(f"Downloaded {success_count}/{len(dataset_keys)} datasets")
        return success_count > 0


# ============================================================================
# MODULE VERIFICATION
# ============================================================================

def verify_modules():
    """Verify that all project modules can be imported."""
    print_step(5, 6, "Verifying Project Modules")
    
    modules_to_check = [
        ('ga.genetic_algorithm', 'Genetic Algorithm'),
        ('ga.unified_optimizer', 'Unified Optimizer'),
        ('ga.multi_objective', 'Multi-Objective (NSGA-II)'),
        ('preprocessing.data_loader', 'Data Loader'),
        ('preprocessing.missing_values', 'Missing Values Handler'),
        ('preprocessing.scaling', 'Feature Scaling'),
        ('preprocessing.smote_handler', 'SMOTE Handler'),
        ('preprocessing.pca', 'PCA'),
        ('models.xgboost_optimizer', 'XGBoost Optimizer'),
        ('models.lightgbm_optimizer', 'LightGBM Optimizer'),
        ('models.neural_network_optimizer', 'Neural Network Optimizer'),
        ('models.svm_optimizer', 'SVM Optimizer'),
        ('utils.logger', 'Logger'),
        ('utils.colors', 'Terminal Colors'),
        ('utils.metrics', 'Metrics Calculator'),
        ('utils.visualization', 'Visualization'),
    ]
    
    success = 0
    failed = []
    
    for module_name, display_name in modules_to_check:
        try:
            importlib.import_module(module_name)
            print_success(f"{display_name:35} OK")
            success += 1
        except Exception as e:
            print_error(f"{display_name:35} FAILED: {e}")
            failed.append((module_name, str(e)))
    
    print()
    
    if failed:
        print_warning(f"{len(failed)} module(s) failed to load")
        print()
        print("Failed modules:")
        for module, error in failed:
            print(f"  - {module}: {error}")
        return False
    else:
        print_success(f"All {success} modules verified")
        return True


# ============================================================================
# QUICK TEST
# ============================================================================

def run_quick_test():
    """Run a quick optimization test."""
    print_step(6, 6, "Running Quick Test")
    
    try:
        from preprocessing.data_loader import DatasetLoader
        from ga.unified_optimizer import UnifiedOptimizer
        from ga.genetic_algorithm import GAConfig
        import numpy as np
        
        print_info("Creating synthetic test dataset...")
        
        # Create small synthetic dataset
        np.random.seed(42)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        print_success(f"Test dataset:  {X.shape[0]} samples, {X.shape[1]} features")
        
        print()
        print_info("Running quick GA optimization (5 individuals, 2 generations)...")
        print()
        
        # Quick optimization
        ga_config = GAConfig(
            population_size=5,
            num_generations=2,
            verbose=0  # Silent
        )
        
        optimizer = UnifiedOptimizer(
            X, y,
            model_type='random_forest',
            fitness_metrics='accuracy',
            ga_config=ga_config,
            logger=None
        )
        
        results = optimizer.optimize()
        
        print()
        print_success(f"Test completed!  CV Score: {results['best_score']:.4f}")
        print()
        
        return True
        
    except Exception as e: 
        print_error(f"Test failed: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# SYSTEM DIAGNOSTICS
# ============================================================================

def print_system_info():
    """Print system diagnostics."""
    print_header("SYSTEM DIAGNOSTICS")
    
    import platform
    
    print(f"  Operating System: {platform.system()} {platform.release()}")
    print(f"  Python Version:    {sys.version.split()[0]}")
    print(f"  Python Path:      {sys.executable}")
    print(f"  Working Directory: {Path.cwd()}")
    
    # Check for GPU (optional)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU Available:     Yes ({torch.cuda.get_device_name(0)})")
        else:
            print(f"  GPU Available:    No (CPU only)")
    except ImportError:
        print(f"  GPU Available:    N/A (PyTorch not installed)")
    
    print()


# ============================================================================
# MAIN SETUP FUNCTION
# ============================================================================

def main():
    """Main setup routine."""
    
    print_header("GA HYPERPARAMETER OPTIMIZATION - SETUP & VERIFICATION")
    
    print("This script will:")
    print("  1. Check Python version")
    print("  2. Verify dependencies")
    print("  3. Set up Kaggle API")
    print("  4. Download datasets")
    print("  5. Verify project modules")
    print("  6. Run a quick test")
    print()
    
    response = input("Continue with setup? [Y/n]: ").strip().lower()
    
    if response == 'n':
        print_info("Setup cancelled")
        return
    
    # Track results
    checks = {
        'python_version':  False,
        'dependencies': False,
        'kaggle':  False,
        'datasets': False,
        'modules': False,
        'test': False
    }
    
    # Run checks
    print_system_info()
    
    checks['python_version'] = check_python_version()
    
    if checks['python_version']: 
        checks['dependencies'] = check_dependencies()
    
    if checks['dependencies']:
        checks['kaggle'] = check_kaggle_setup()
        checks['datasets'] = check_datasets()
        checks['modules'] = verify_modules()
        
        if checks['modules']:
            response = input("\nRun quick test? [Y/n]: ").strip().lower()
            if response != 'n':
                checks['test'] = run_quick_test()
    
    # Summary
    print_header("SETUP SUMMARY")
    
    print(f"  Python Version:   {'✓' if checks['python_version'] else '✗'}")
    print(f"  Dependencies:    {'✓' if checks['dependencies'] else '✗'}")
    print(f"  Kaggle API:      {'✓' if checks['kaggle'] else '⚠'}")
    print(f"  Datasets:        {'✓' if checks['datasets'] else '⚠'}")
    print(f"  Project Modules: {'✓' if checks['modules'] else '✗'}")
    print(f"  Quick Test:      {'✓' if checks['test'] else '⚠'}")
    print()
    
    if all([checks['python_version'], checks['dependencies'], checks['modules']]):
        print_success("Setup complete!  You're ready to run optimizations.")
        print()
        print("Next steps:")
        print("  1. Configure your experiment in main.py")
        print("  2. Run: python main.py")
        print()
        print("Documentation:")
        print("  - Quick Start:   QUICKSTART.md")
        print("  - Full Guide:   README.md")
        print("  - Kaggle Setup: docs/KAGGLE_SETUP.md")
    else:
        print_warning("Setup incomplete. Please resolve issues above.")
        
        if not checks['kaggle']:
            print_info("Note: Kaggle API is optional if datasets are already downloaded")
    
    print()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_info("Setup interrupted by user")
        sys.exit(0)
    except Exception as e: 
        print()
        print_error(f"Setup failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)