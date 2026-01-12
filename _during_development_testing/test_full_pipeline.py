# ============================================================================
# FULL PIPELINE INTEGRATION TEST
# ============================================================================
# Test the complete preprocessing pipeline end-to-end
#
# Pipeline:
#   1. Load dataset
#   2. EDA analysis
#   3. Handle missing values
#   4. Scale features
#   5. Balance classes (SMOTE)
#   6. Reduce dimensions (PCA)
#   7. Train baseline models
#   8. Evaluate performance
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import time
import json

from preprocessing.data_loader import DatasetLoader
from preprocessing.eda import EDA
from preprocessing.missing_values import MissingValuesHandler
from preprocessing.scaling import StandardScaler, MinMaxScaler, RobustScaler
from preprocessing.smote_handler import SMOTEHandler
from preprocessing.pca import PCA

from utils.logger import get_logger
from utils.colors import print_header, print_section, print_success, print_info, print_warning

# ML models
try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print_warning("scikit-learn not available. Model training will be skipped.")


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'dataset':  'secom',
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    
    'preprocessing': {
        'missing_values_strategy': 'mean',
        'scaler_type': 'standard',
        'smote_strategy': 'smote',
        'pca_variance':  0.95
    },
    
    'models': {
        'svm':  {
            'name': 'Support Vector Machine',
            'params': {'C': 1.0, 'kernel': 'rbf', 'random_state': 42}
        },
        'rf': {
            'name': 'Random Forest',
            'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
        },
        'knn': {
            'name': 'K-Nearest Neighbors',
            'params': {'n_neighbors': 5}
        },
        'nb':  {
            'name': 'Naive Bayes',
            'params': {}
        }
    }
}


# ============================================================================
# MAIN INTEGRATION TEST
# ============================================================================

def main():
    """Run full pipeline integration test."""
    
    logger = get_logger(name="INTEGRATION_TEST", verbose=True)
    
    print_header("FULL PIPELINE INTEGRATION TEST")
    print()
    
    results = {
        'config': CONFIG,
        'timing': {},
        'data_stats': {},
        'model_results': {}
    }
    
    # Create output directory
    output_dir = Path('outputs/integration_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load Dataset
    # ========================================================================
    
    print_section("STEP 1: Load Dataset")
    
    start_time = time.time()
    
    loader = DatasetLoader(logger=logger, interactive=False)
    X, y, metadata = loader.load_dataset(CONFIG['dataset'])
    
    results['timing']['load_dataset'] = time.time() - start_time
    results['data_stats']['original'] = {
        'shape': X.shape,
        'samples': int(X.shape[0]),
        'features': int(X.shape[1]),
        'classes': int(len(np.unique(y))),
        'missing_values': int(np.isnan(X).sum()),
        'missing_percentage': float(np.isnan(X).sum() / X.size * 100)
    }
    
    logger.blank()
    print_info(f"Dataset loaded:  {X.shape}")
    print_info(f"Missing values: {results['data_stats']['original']['missing_values']} ({results['data_stats']['original']['missing_percentage']:.2f}%)")
    logger.blank()
    
    # ========================================================================
    # STEP 2: EDA Analysis
    # ========================================================================
    
    print_section("STEP 2: Exploratory Data Analysis")
    
    start_time = time.time()
    
    eda = EDA(X, y, logger=logger)
    
    # Generate quick report (no visualizations for speed)
    print_info("Running statistical analysis...")
    stats = eda.get_statistics()
    corr_pairs = eda.find_highly_correlated_features(threshold=0.9)
    
    results['timing']['eda'] = time.time() - start_time
    results['data_stats']['eda'] = {
        'constant_features': int((stats['std'] == 0).sum()),
        'highly_correlated_pairs': len(corr_pairs),
        'features_with_missing': int((stats['missing'] > 0).sum())
    }
    
    print_info(f"  Constant features: {results['data_stats']['eda']['constant_features']}")
    print_info(f"  Highly correlated pairs: {results['data_stats']['eda']['highly_correlated_pairs']}")
    
    logger.blank()
    
    # ========================================================================
    # STEP 3: Handle Missing Values
    # ========================================================================
    
    print_section("STEP 3: Handle Missing Values")
    
    start_time = time.time()
    
    mv_handler = MissingValuesHandler(
        strategy=CONFIG['preprocessing']['missing_values_strategy'],
        logger=logger
    )
    
    X_clean = mv_handler.fit_transform(X)
    
    results['timing']['missing_values'] = time.time() - start_time
    results['data_stats']['after_imputation'] = {
        'shape': X_clean.shape,
        'missing_values': int(np.isnan(X_clean).sum())
    }
    
    print_success(f"✓ Missing values handled: {np.isnan(X).sum()} → {np.isnan(X_clean).sum()}")
    
    logger.blank()
    
    # ========================================================================
    # STEP 4: Scale Features
    # ========================================================================
    
    print_section("STEP 4: Scale Features")
    
    start_time = time.time()
    
    scaler_type = CONFIG['preprocessing']['scaler_type']
    
    if scaler_type == 'standard':
        scaler = StandardScaler(logger=logger)
    elif scaler_type == 'minmax': 
        scaler = MinMaxScaler(logger=logger)
    elif scaler_type == 'robust':
        scaler = RobustScaler(logger=logger)
    else:
        raise ValueError(f"Unknown scaler:  {scaler_type}")
    
    X_scaled = scaler.fit_transform(X_clean)
    
    results['timing']['scaling'] = time.time() - start_time
    results['data_stats']['after_scaling'] = {
        'mean': float(X_scaled.mean()),
        'std': float(X_scaled.std()),
        'min': float(X_scaled.min()),
        'max': float(X_scaled.max())
    }
    
    print_success(f"✓ Features scaled using {scaler_type}")
    print_info(f"  Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
    
    logger.blank()
    
    # ========================================================================
    # STEP 5: Balance Classes (SMOTE)
    # ========================================================================
    
    print_section("STEP 5: Balance Classes (SMOTE)")
    
    start_time = time.time()
    
    smote_handler = SMOTEHandler(
        strategy=CONFIG['preprocessing']['smote_strategy'],
        random_state=CONFIG['random_state'],
        logger=logger
    )
    
    # Get class distribution before
    unique_before, counts_before = np.unique(y, return_counts=True)
    
    X_balanced, y_balanced = smote_handler.fit_resample(X_scaled, y)
    
    # Get class distribution after
    unique_after, counts_after = np.unique(y_balanced, return_counts=True)
    
    results['timing']['smote'] = time.time() - start_time
    results['data_stats']['after_smote'] = {
        'shape': X_balanced.shape,
        'class_distribution_before': {int(k): int(v) for k, v in zip(unique_before, counts_before)},
        'class_distribution_after': {int(k): int(v) for k, v in zip(unique_after, counts_after)}
    }
    
    print_success(f"✓ Classes balanced: {X_scaled.shape[0]} → {X_balanced.shape[0]} samples")
    print_info(f"  Before: {dict(zip(unique_before, counts_before))}")
    print_info(f"  After:   {dict(zip(unique_after, counts_after))}")
    
    logger.blank()
    
    # ========================================================================
    # STEP 6: Reduce Dimensions (PCA)
    # ========================================================================
    
    print_section("STEP 6: Reduce Dimensions (PCA)")
    
    start_time = time.time()
    
    pca = PCA(
        n_components=CONFIG['preprocessing']['pca_variance'],
        random_state=CONFIG['random_state'],
        logger=logger
    )
    
    X_pca = pca.fit_transform(X_balanced)
    
    results['timing']['pca'] = time.time() - start_time
    results['data_stats']['after_pca'] = {
        'shape': X_pca.shape,
        'original_features': int(X_balanced.shape[1]),
        'reduced_features': int(X_pca.shape[1]),
        'reduction_percentage': float((1 - X_pca.shape[1] / X_balanced.shape[1]) * 100),
        'variance_explained': float(pca.explained_variance_ratio_.sum() * 100)
    }
    
    print_success(f"✓ Dimensions reduced:  {X_balanced.shape[1]} → {X_pca.shape[1]} features")
    print_info(f"  Variance explained: {pca.explained_variance_ratio_.sum() * 100:.2f}%")
    print_info(f"  Reduction: {(1 - X_pca.shape[1] / X_balanced.shape[1]) * 100:.1f}%")
    
    logger.blank()
    
    # ========================================================================
    # STEP 7: Prepare Train/Test Split
    # ========================================================================
    
    print_section("STEP 7: Train/Test Split")
    
    if not SKLEARN_AVAILABLE:
        print_warning("scikit-learn not available. Skipping model training.")
        save_results(results, output_dir)
        return
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_balanced,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y_balanced
    )
    
    results['data_stats']['split'] = {
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'train_ratio': float(len(X_train) / len(X_pca)),
        'test_ratio': float(len(X_test) / len(X_pca))
    }
    
    print_info(f"Train:  {X_train.shape}, Test: {X_test.shape}")
    
    logger.blank()
    
    # ========================================================================
    # STEP 8: Train Baseline Models
    # ========================================================================
    
    print_section("STEP 8: Train Baseline Models")
    print()
    
    for model_key, model_config in CONFIG['models'].items():
        logger.section(f"Model: {model_config['name']}")
        
        start_time = time.time()
        
        # Create model
        if model_key == 'svm':
            model = SVC(**model_config['params'])
        elif model_key == 'rf':
            model = RandomForestClassifier(**model_config['params'])
        elif model_key == 'knn': 
            model = KNeighborsClassifier(**model_config['params'])
        elif model_key == 'nb':
            model = GaussianNB(**model_config['params'])
        else:
            continue
        
        # Train
        logger.info("Training...")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation
        logger.info("Running cross-validation...")
        cv = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=CONFIG['random_state'])
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        training_time = time.time() - start_time
        
        # Store results
        results['model_results'][model_key] = {
            'name': model_config['name'],
            'params': model_config['params'],
            'test_accuracy': float(accuracy),
            'test_f1_score': float(f1),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'cv_scores': [float(s) for s in cv_scores],
            'training_time': float(training_time)
        }
        
        print_success(f"✓ {model_config['name']} trained")
        print_info(f"  Test Accuracy:   {accuracy:.4f}")
        print_info(f"  Test F1-Score:  {f1:.4f}")
        print_info(f"  CV Accuracy:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print_info(f"  Training time:  {training_time:.2f}s")
        
        logger.blank()
    
    # ========================================================================
    # STEP 9: Summary & Results
    # ========================================================================
    
    print_header("INTEGRATION TEST SUMMARY")
    print()
    
    # Timing summary
    print_section("Timing Summary")
    total_time = sum(results['timing'].values())
    
    for step, duration in results['timing'].items():
        pct = (duration / total_time) * 100
        print(f"  {step:20} {duration:>8.3f}s ({pct: > 5.1f}%)")
    
    print()
    print_info(f"Total time: {total_time:.2f}s")
    
    logger.blank()
    
    # Model comparison
    print_section("Model Performance Comparison")
    print()
    
    print(f"{'Model':<25} {'Test Acc': >10} {'CV Acc':>15} {'F1-Score':>10} {'Time': >10}")
    print("-" * 80)
    
    for model_key, model_results in results['model_results'].items():
        name = model_results['name']
        test_acc = model_results['test_accuracy']
        cv_acc = model_results['cv_accuracy_mean']
        cv_std = model_results['cv_accuracy_std']
        f1 = model_results['test_f1_score']
        train_time = model_results['training_time']
        
        print(f"{name:<25} {test_acc:>10.4f} {cv_acc:>7.4f}±{cv_std: <5.4f} {f1:>10.4f} {train_time:>9.2f}s")
    
    logger.blank()
    
    # Best model
    best_model_key = max(results['model_results'].items(), 
                         key=lambda x:  x[1]['cv_accuracy_mean'])[0]
    best_model = results['model_results'][best_model_key]
    
    print_success(f"Best model: {best_model['name']}")
    print_info(f"  CV Accuracy:  {best_model['cv_accuracy_mean']:.4f} ± {best_model['cv_accuracy_std']:.4f}")
    print_info(f"  Test Accuracy: {best_model['test_accuracy']:.4f}")
    
    logger.blank()
    
    # Save results
    save_results(results, output_dir)
    
    print_success(f"✓ Integration test complete!")
    print_info(f"  Results saved to: {output_dir}")


def save_results(results:  dict, output_dir: Path):
    """Save results to JSON file."""
    
    output_file = output_dir / 'integration_test_results.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print_info(f"Results saved to: {output_file}")


# ============================================================================
# RUN TEST
# ============================================================================

if __name__ == '__main__':
    main()