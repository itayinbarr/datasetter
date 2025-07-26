# EEG Dataset Preprocessing & Unification

**Author:** Itay  
**Version:** 0.0.1

## Overview

This repository provides a Python-based pipeline for unifying and preprocessing EEG data from multiple OpenNeuro datasets. By leveraging MNE-Python and PyTorch, the code automatically:

- Reads raw EEG data from selected (supported) OpenNeuro datasets.
- Resamples the data to a common sample rate.
- Segments the data into epochs of uniform duration.
- Aligns channels to a standardized set of common EEG electrode names.

The goal is to simplify the use of multiple OpenNeuro datasets that have different recording setups so that researchers can combine them into a single, uniform dataset without wrestling with dataset-specific quirks.

## Why OpenNeuro?

[OpenNeuro](https://openneuro.org/) is a free and open platform for sharing MRI, EEG, iEEG, and MEG data. We chose to support OpenNeuro because:

- It is open source and encourages reproducible science.
- Datasets are freely accessible to anyone.
- It has a large variety of EEG studies, enabling broader generalization.

## Supported Datasets

As of now, the following dataset IDs are recognized in the code (`recognized_datasets` list in `EEGDataset`):

1. **ds002691** - [OpenNeuro](https://openneuro.org/datasets/ds002691)
2. **ds002778** - [OpenNeuro](https://openneuro.org/datasets/ds002778)
3. **ds003800** - [OpenNeuro](https://openneuro.org/datasets/ds003800)
4. **ds005420** - [OpenNeuro](https://openneuro.org/datasets/ds005420)

> **Note:** If you attempt to load a dataset not in this list, the code will raise an error. We plan to add more datasets soon. Feel free to contribute by following our [CONTRIBUTING guidelines](./CONTRIBUTING.md).

## Installation & Requirements

1. **Python 3.8+** (recommended)
2. **MNE-Python** (e.g., `pip install mne`)
3. **PyTorch** (CPU or GPU version depending on your environment)
4. **NumPy, pandas, matplotlib**, and **scipy** for data manipulation and plotting.

Alternatively, install all dependencies via:

```bash
pip install -r requirements.txt
```

_(Adjust or create `requirements.txt` as you see fit.)_

## How to Download Data from OpenNeuro

You must download the raw data from OpenNeuro **before** running our pipeline. We suggest using the official OpenNeuro CLI:

```bash
npm install --global @openneuro/cli
openneuro download --snapshot <snapshot_number> <dataset_id> <target_directory>
```

For example, to download `ds002691`

```bash
openneuro download --snapshot 1.0.1 ds002691 ds002691-download/
```

**Make sure** that when you download `ds00XXXX`, you place it in the `data/raw` directory (or another location that you configure in your code).

For more details on other methods (AWS CLI, Datalad, browser downloads), see [OpenNeuro’s official docs](https://openneuro.org/).

## Usage

1. **Clone the repository** and enter the directory:

   ```bash
   git clone https://github.com/happy-thalamus/datasetter.git
   cd datasetter
   ```

2. **Download** the supported OpenNeuro datasets (see above). Place them in `./data/raw/` (or wherever you prefer).
3. **Run** the example script:

   ```bash
   python examples/data_preparation.py
   ```

   This script will:

   - Detect if a processed dataset already exists in `./data/processed/`.
   - If not, create one by loading, resampling, epoching, and saving the unified data.
   - Generate example plots in `./data/processed/plots/`.

For more detailed showcases, see [docs/showcase.md](./docs/showcase.md).

## Planned Enhancements

- **Additional Dataset Loaders**: We plan to add more datasets from OpenNeuro as well as from other open EEG repositories.
- **Better Channel Selection**: We want to add a more flexible interface for custom channel subsets.
- **Artifact Rejection**: Automatic artifact detection and rejection (e.g., using ICA) is in development.
- **More Visualizations**: We aim to produce more advanced interactive plots for quick data QA.

## Contributing

We welcome contributions and new dataset support! Please read our [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on filing issues, creating pull requests, or adding new dataset loaders.

## License

[MIT License](./LICENSE)

---

_Happy EEG analyzing!_

# datasetter
