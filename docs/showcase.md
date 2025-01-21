# Usage Showcase

This document provides a more detailed walkthrough of how to use `EEGDataset`.

---

## 1. Quick Start

```python
from src.data.dataset import EEGDataset

# Suppose you've already downloaded ds003800 and ds002778 into 'data/raw'
dataset_ids = ['ds003800', 'ds002778']
data_dir = 'data/raw'
output_dir = 'data/processed'

# Create the dataset
dataset = EEGDataset(
    data_dir=data_dir,
    dataset_ids=dataset_ids,
    output_dir=output_dir,
    target_sample_rate=250,   # Adjust as needed
    epoch_duration=4.0        # e.g., 4-second epochs
)

# Save the processed dataset
dataset.save_dataset(output_dir)

# Basic details
print("Number of epochs:", len(dataset))
print("Channels used:", dataset.standard_channels)
```

---

## 2. Loading a Saved Dataset

After you’ve processed data once, you might want to just load it in subsequent scripts:

```python
# If you've previously saved the dataset:
saved_dataset = EEGDataset.load_saved_dataset('data/processed')

print("Loaded dataset with", len(saved_dataset), "epochs")
print("Label distribution:", set(saved_dataset.string_labels))
```

---

## 3. Plotting & Visualization

The class offers helper functions for plotting:

```python
# Plot a single epoch
index = 10  # random example
saved_dataset.plot_example(index)

# Plot multiple random examples
saved_dataset.plot_random_examples(n=3)
```

---

## 4. Adding New Datasets

1. **Create a private method** in `EEGDataset` (e.g., `_load_myNewDataset`).
2. **Add** your dataset identifier to `recognized_datasets`.
3. **Update** `load_data()` to call your loader method for the new identifier.
4. **Test** by downloading your dataset to `data/raw` and verifying the shapes and labels.

---

## 5. Common Issues / FAQ

- **Error: `FileNotFoundError: Dataset directory X does not exist.`**  
   Make sure you placed the downloaded dataset in the correct folder with the naming convention `<dataset_id>-download`. For example, `ds003800-download`.
- **Mismatch in channel names**:  
   Some data might have unexpected or non-standard channel naming. Consider editing `_standardize_channel_names` or checking the raw data.

For more advanced usage or troubleshooting, check the code docstrings or reach out via [Issues](https://github.com/YourName/my_eeg_project/issues).

---

Enjoy analyzing your EEG data!
