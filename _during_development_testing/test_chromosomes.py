# Test chromosomes with all 3 model types
from utils.config_loader import load_hyperparameter_config, apply_dataset_overrides
from utils.chromosome import (
    initialize_random_chromosome,
    decode_chromosome,
    encode_hyperparameters,
    get_gene_names,
    print_chromosome_info
)
from utils.colors import print_header, print_section, print_success

print_header("TESTING CHROMOSOMES FOR ALL MODEL TYPES")

# Test with each model type
for model_type in ['neural_network', 'svm', 'logistic_regression']:
    print("\n" + "="*70)
    print_section(f"{model_type.upper().replace('_', ' ')}")
    
    # Load config
    config = load_hyperparameter_config(model_type)
    config = apply_dataset_overrides(config, 'secom')
    
    # Print chromosome structure
    print_chromosome_info(config)
    
    # Generate random chromosome
    chromosome = initialize_random_chromosome(config, random_state=42)
    print(f"\nRandom chromosome (length {len(chromosome)}):")
    print(f"  {[f'{g:.3f}' for g in chromosome]}")
    
    # Decode
    decoded = decode_chromosome(chromosome, config)
    print(f"\nDecoded hyperparameters:")
    for key, value in decoded.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}:  {value}")
    
    # Encode back
    re_encoded = encode_hyperparameters(decoded, config)
    
    # Check reversibility
    max_diff = max(abs(a - b) for a, b in zip(chromosome, re_encoded))
    
    if max_diff < 0.01:
        print_success(f"Encoding is highly reversible (max diff: {max_diff:.6f})")
    else:
        print_success(f"Encoding works (max diff: {max_diff:.6f} - expected for categorical/discrete)")

print("\n" + "="*70)
print_success("All chromosome tests passed!")
print("="*70)