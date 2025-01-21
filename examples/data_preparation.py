"""
This script demonstrates how to use the EEGDataset class to easily create and process an EEG dataset.

It shows the following key steps:
1. Loading raw data from multiple datasets
2. Preprocessing the data (resampling, epoching)
3. Saving and loading the processed dataset
4. Visualizing sample data from each dataset
5. Printing summary statistics of the processed data

This serves as a practical example for researchers and developers working with EEG data,
showcasing the simplicity and efficiency of using the EEGDataset class for data preparation.
"""

# data_preparation.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Append path to sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import EEGDataset

# Configuration
data_dir = './data/raw/'
output_dir = './data/processed/'
processed_data_dir = './data/processed/'
dataset_ids = ['ds003800', 'ds002778', 'ds002691', 'ds005420']
target_sample_rate = 125  # Hz
epoch_duration = 8.0  # seconds

# Check if processed dataset already exists
if os.path.exists(os.path.join(processed_data_dir, 'eeg_data.npy')):
    print("Processed dataset found. Loading existing dataset...")
    dataset = EEGDataset.load_saved_dataset(processed_data_dir)
else:
    print("Processed dataset not found. Creating new dataset...")
    # Data Loading and Preprocessing using EEGDataset
    dataset = EEGDataset(
        data_dir=data_dir,
        output_dir=output_dir,
        dataset_ids=dataset_ids,
        transform=None,
        target_sample_rate=target_sample_rate,
        epoch_duration=epoch_duration
    )

    # Save the processed dataset
    dataset.save_dataset(processed_data_dir)

print(f"Dataset loaded. Number of samples: {len(dataset)}")
print("Unique string labels:", set(dataset.string_labels))

# Create a new directory for plots
plots_dir = './data/processed/plots'
os.makedirs(plots_dir, exist_ok=True)

# Save examples from each dataset
for dataset_id in dataset_ids:
    dataset.save_dataset_examples(dataset_id, plots_dir)

print("Data preparation and visualization complete.")

# Print some sample data for verification
print("\nSample data:")
for i in range(min(5, len(dataset))):
    data, label = dataset[i]
    print(f"Sample {i}:")
    print(f"  Shape: {data.shape}")
    print(f"  Label: {label} ({dataset.num_to_label[label]})")
    print(f"  Data range: {data.min().item():.2f} to {data.max().item():.2f}")
    print(f"  Mean: {data.mean().item():.2f}")
    print(f"  Std: {data.std().item():.2f}")
