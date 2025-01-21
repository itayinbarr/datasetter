# Contributing to the EEG Dataset Preprocessing Project

We’re excited you want to contribute! Here are some guidelines to make the process smoother.

## Ways to Contribute

1. **Add a new dataset loader**:
   - Check `EEGDataset.recognized_datasets` to see if your dataset is already supported.
   - If not, create a new method (e.g., `_load_<dataset_id>`) that handles the dataset’s file naming, events, or any special preprocessing steps.
   - Add your dataset ID to `self.recognized_datasets` and point the `load_data()` method to call your new loader method when that dataset ID is encountered.
2. **Fix bugs** or **improve performance**: Submit an issue describing the bug or inefficiency, then propose a fix.
3. **Improve documentation**: If you notice any missing or unclear instructions, feel free to submit a PR updating `README.md`, the docstring in `dataset.py`, or any markdown files in `docs/`.
4. **Add new features** (like artifact rejection, more advanced transformations, etc.):
   - Create an issue or discussion to propose your idea.
   - Open a PR with your changes once there’s consensus.

## Development Setup

1. **Fork** this repo and clone your fork locally.
2. Create a Python virtual environment or Conda environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

```

4. You can run the example script in `examples/data_preparation.py` to ensure everything works.

## Submitting Pull Requests

1. **Branch naming**: name your branch something descriptive, like `feature/ds999999-loader` or `fix/channel-name-bug`.
2. **Pull Request description**: Please include:
    - A reference to the issue number (if applicable).
    - A short description of what you changed and why.
    - Any additional details to help reviewers understand your approach.
3. Ensure all relevant tests pass (if you added or updated tests), and that your code adheres to Python best practices (PEP8 formatting is always a plus).

## Code of Conduct

Please be respectful and constructive in all communications. We appreciate all contributions and aim to foster a welcoming environment for everyone.

Thank you for helping improve this project!
```
