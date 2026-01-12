from preprocessing. data_loader import DatasetLoader
from ga.ml_optimizer import MLOptimizer
from ga.genetic_algorithm import GAConfig

loader = DatasetLoader(interactive=False)
X, y, _ = loader.load_dataset('secom')

# Baseline config from integration test
baseline_config = {
    'n_estimators': 100,
    'max_depth':  10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'pca_variance': 0.95,
    'smote_strategy': 'smote',
    'scaler': 'standard'
}

ga_config = GAConfig(population_size=1, num_generations=1, cache_fitness=False)
optimizer = MLOptimizer(X, y, 'random_forest', ga_config, cv_folds=3)

print("Testing baseline config...")
score = optimizer._evaluate_configuration(baseline_config)
print(f"Score: {score:.4f}")
print(f"Expected: ~0.99")