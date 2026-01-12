# Create a sample custom dataset for testing

import numpy as np
import pandas as pd
from pathlib import Path

print("Creating sample custom dataset...")

# Generate synthetic data
np.random.seed(42)

n_samples = 500
n_features = 10

X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, 3, n_samples)  # 3 classes

# Create DataFrame
feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['label'] = y

# Save to CSV
output_dir = Path('datasets/custom/my_test_dataset')
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / 'data.csv'
df.to_csv(output_file, index=False)

print(f"✓ Created:  {output_file}")
print(f"  Samples: {n_samples}")
print(f"  Features: {n_features}")
print(f"  Classes:  3")
print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")