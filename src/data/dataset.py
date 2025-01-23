# dataset.py
import os
import sys
import contextlib
import numpy as np
import torch
from torch.utils.data import Dataset
import mne
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy.signal import butter, filtfilt

class EEGDataset(Dataset):
    """
    A PyTorch Dataset for loading and preprocessing EEG data from multiple datasets.

    This dataset class supports loading data from different EEG datasets, resampling to a target
    sample rate, segmenting into epochs, and standardizing channel selection.

    Attributes:
        transform (callable, optional): Optional transform to be applied on a sample.
        target_sample_rate (int): The desired sample rate for all EEG data.
        dataset_ids (list): List of dataset identifiers to load.
        epoch_duration (float): Duration of each epoch in seconds.
        epoch_samples (int): Number of samples in each epoch.
        recognized_datasets (list): List of supported dataset identifiers.
        standard_channels (list): List of standard EEG channel names.
        data (list): List of preprocessed EEG epochs.
        labels (list): List of labels corresponding to each epoch.
    """

    def __init__(self, data_dir, dataset_ids, output_dir, transform=None, target_sample_rate=250, epoch_duration=4.0):
        """
        Initialize the EEGDataset.

        Args:
            data_dir (str): Root directory containing the datasets.
            dataset_ids (list): List of dataset identifiers to load.
            output_dir (str): Directory to save the processed dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_sample_rate (int): The desired sample rate for all EEG data.
            epoch_duration (float): Duration of each epoch in seconds.
        """
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.dataset_ids = dataset_ids
        self.epoch_duration = epoch_duration
        self.output_dir = output_dir
        self.epoch_samples = int(self.target_sample_rate * self.epoch_duration)
        self.recognized_datasets = ['ds002691', 'ds002778', 'ds003800', 'ds005420']  # Add more dataset IDs here as needed
        for dataset_id in self.dataset_ids:
            if dataset_id not in self.recognized_datasets:
                raise ValueError(f"Dataset ID '{dataset_id}' is not recognized.")
        self.standard_channels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2', 'Fpz', 'AF3', 'AF4', 'F5',
            'F1', 'F2', 'F6', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10'
        ]
        self.important_channels = [
            'Fz', 'Cz', 'Pz',  # Midline
            'F3', 'F4', 'C3', 'C4', 'P3', 'P4',  # Left-right pairs
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6',  # Temporal
            'Fp1', 'Fp2', 'O1', 'O2',  # Frontal pole and occipital
            'Fpz', 'Oz',  # Additional midline
            'AF3', 'AF4', 'F5', 'F6', 'FC5', 'FC6',  # Frontal and fronto-central
            'FC1', 'FC2', 'FT10'  # Additional channels
        ]
        self.data, self.string_labels = self.load_data(data_dir)
        self.data = self.standardize_epochs(self.data)
        self.label_to_num, self.num_to_label = self.create_label_mappings()
        self.labels = [self.label_to_num[label] for label in self.string_labels]
        
        

    def load_data(self, data_dir):
        """
        Load data from all specified datasets.

        Args:
            data_dir (str): Root directory containing the datasets.

        Returns:
            tuple: (list of EEG epochs, list of corresponding labels)
        """
        data = []
        labels = []
        for dataset_id in self.dataset_ids:
            if dataset_id == 'ds002691':
                dataset_data, dataset_labels = self._load_ds002691(data_dir, dataset_id)
            elif dataset_id == 'ds002778':
                dataset_data, dataset_labels = self._load_ds002778(data_dir, dataset_id)
            elif dataset_id == 'ds003800':
                dataset_data, dataset_labels = self._load_ds003800(data_dir, dataset_id)
            elif dataset_id == 'ds005420':
                dataset_data, dataset_labels = self._load_ds005420(data_dir, dataset_id)
            else:
                raise NotImplementedError(f"Dataset ID '{dataset_id}' is not implemented.")
            
            # Add dataset_id to labels
            dataset_labels = [f"{dataset_id}_{label}" for label in dataset_labels]
            
            data.extend(dataset_data)
            labels.extend(dataset_labels)
            
            # Print some statistics about the loaded data
            print(f"Loaded {len(dataset_data)} samples from {dataset_id}")
            print(f"Sample data shape: {dataset_data[0].shape}")
            print(f"Sample data range: {np.min(dataset_data[0])} to {np.max(dataset_data[0])}")
            print(f"Sample data mean: {np.mean(dataset_data[0])}")
            print(f"Sample data std: {np.std(dataset_data[0])}")
            print(f"Unique labels: {set(dataset_labels)}")
            print()

        print(f"Data loading complete. Total epochs: {len(data)}")
        return data, labels

    @contextlib.contextmanager
    def suppress_output(self):
        """
        A context manager to temporarily suppress stdout and stderr.
        """
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    def _load_ds002691(self, data_dir, dataset_id):
        """
        Load data from the ds002691 dataset.

        Args:
            data_dir (str): Root directory containing the datasets.
            dataset_id (str): Identifier for the ds002691 dataset.

        Returns:
            tuple: (list of EEG epochs, list of corresponding labels)
        """
        dataset_data = []
        dataset_labels = []
        dataset_dir = os.path.join(data_dir, dataset_id + '-download')
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
        subject_count = 0
        for sub_folder in os.listdir(dataset_dir):
            if sub_folder == 'code':
                continue
            sub_folder_path = os.path.join(dataset_dir, sub_folder)
            if os.path.isdir(sub_folder_path):
                eeg_subfolder_path = os.path.join(sub_folder_path, 'eeg')
                if not os.path.exists(eeg_subfolder_path):
                    raise FileNotFoundError(f"EEG subfolder '{eeg_subfolder_path}' does not exist.")
                
                for filename in os.listdir(eeg_subfolder_path):
                    if filename.endswith('_eeg.set'):
                        subject_count += 1
                        filepath = os.path.join(eeg_subfolder_path, filename)
                        with self.suppress_output():
                            raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
                            raw.resample(self.target_sample_rate, verbose=False)
                        raw = self._match_standard_channels(raw)

                        # Read events.tsv
                        events_tsv = os.path.join(eeg_subfolder_path, filename.replace('_eeg.set', '_events.tsv'))
                        events_df = pd.read_csv(events_tsv, sep='\t')

                        # Map trial_type to labels
                        event_id = {'wait': 'Wait', 'relax': 'Relax', 'getready': 'GetReady', 'concentrate': 'Concentrate'}

                        # Create events array
                        events = []
                        for idx, row in events_df.iterrows():
                            onset = row['onset']
                            trial_type = row['trial_type'].lower()
                            if trial_type in event_id:
                                event_time = int(onset * self.target_sample_rate)
                                event_code = list(event_id.values()).index(event_id[trial_type])
                                events.append([event_time, 0, event_code])
                        events = np.array(events, dtype='int')

                        # Ensure unique event times
                        unique_times, unique_indices = np.unique(events[:, 0], return_index=True)
                        events = events[unique_indices]

                        # Create epochs
                        tmin = 0
                        tmax = self.epoch_duration - 1 / self.target_sample_rate
                        with self.suppress_output():
                            epochs = mne.Epochs(raw, events, event_id={v: k for k, v in enumerate(event_id.values())}, 
                                                tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)

                        data = epochs.get_data()
                        labels = [f"{dataset_id}_{event_id[events_df.iloc[i]['trial_type'].lower()]}" for i in range(len(data))]

                        dataset_data.extend(data)
                        dataset_labels.extend(labels)
        print(f"Finished processing {subject_count} subjects for dataset {dataset_id}")
        return dataset_data, dataset_labels

    def _load_ds002778(self, data_dir, dataset_id):
        """
        Load data from the ds002778 dataset.

        Args:
            data_dir (str): Root directory containing the datasets.
            dataset_id (str): Identifier for the ds002778 dataset.

        Returns:
            tuple: (list of EEG epochs, list of corresponding labels)
        """
        dataset_data = []
        dataset_labels = []
        dataset_dir = os.path.join(data_dir, dataset_id + '-download')
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
        subject_count = 0

        for sub_folder in os.listdir(dataset_dir):
            sub_folder_path = os.path.join(dataset_dir, sub_folder)
            if os.path.isdir(sub_folder_path) and sub_folder.startswith('sub-'):
                if 'hc' in sub_folder:
                    eeg_folder = os.path.join(sub_folder_path, 'eeg')
                    if os.path.exists(eeg_folder):
                        subject_count += 1
                        eeg_file = self._find_bdf_file(eeg_folder, sub_folder)
                        with self.suppress_output():
                            raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)
                            raw = self._process_raw_ds002778(raw)
                            epochs = self._segment_epochs(raw)
                        data = epochs.get_data()
                        dataset_data.extend(data)
                        dataset_labels.extend(['Rest'] * len(data))  # Label 'Rest' for healthy controls
                elif 'pd' in sub_folder:
                    for session in ['ses-on', 'ses-off']:
                        session_folder = os.path.join(sub_folder_path, session, 'eeg')
                        if os.path.exists(session_folder):
                            subject_count += 1
                            eeg_file = self._find_bdf_file(session_folder, sub_folder, session)
                            with self.suppress_output():
                                raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)
                                raw = self._process_raw_ds002778(raw)
                                epochs = self._segment_epochs(raw)
                            data = epochs.get_data()
                            label = 'PD_Rest_On' if session == 'ses-on' else 'PD_Rest_Off'
                            dataset_data.extend(data)
                            dataset_labels.extend([label] * len(data))
        print(f"Finished processing {subject_count} subjects for dataset {dataset_id}")
        return dataset_data, dataset_labels

    def _load_ds003800(self, data_dir, dataset_id):
        """
        Load data from the ds003800 dataset.

        Args:
            data_dir (str): Root directory containing the datasets.
            dataset_id (str): Identifier for the ds003800 dataset.

        Returns:
            tuple: (list of EEG epochs, list of corresponding labels)
        """
        dataset_data = []
        dataset_labels = []
        dataset_dir = os.path.join(data_dir, dataset_id + '-download')
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
        subject_count = 0

        # Loop through subject folders
        for sub_folder in os.listdir(dataset_dir):
            sub_folder_path = os.path.join(dataset_dir, sub_folder)
            if os.path.isdir(sub_folder_path) and sub_folder.startswith('sub-'):
                subject_count += 1
                eeg_folder = os.path.join(sub_folder_path, 'eeg')
                if not os.path.exists(eeg_folder):
                    raise FileNotFoundError(f"EEG folder '{eeg_folder}' does not exist.")

                # Process 'AuditoryGammaEntrainment' task
                task_name = 'AuditoryGammaEntrainment'
                eeg_file = os.path.join(eeg_folder, f"{sub_folder}_task-{task_name}_eeg.set")
                events_file = os.path.join(eeg_folder, f"{sub_folder}_task-{task_name}_events.tsv")
                if os.path.exists(eeg_file) and os.path.exists(events_file):
                    with self.suppress_output():
                        raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
                        raw.resample(self.target_sample_rate, verbose=False)
                    
                    # Handle channel naming issues
                    raw = self._standardize_channel_names(raw)
                    
                    # Try to set montage, but continue if it fails
                    try:
                        raw = self._match_standard_channels(raw)
                    except Exception as e:
                        print(f"Warning: Could not set standard montage for {sub_folder}. Error: {e}")
                        print("Continuing with existing channels.")

                    # Read events from events.tsv
                    events_df = pd.read_csv(events_file, sep='\t')
                    events_sample = []
                    event_labels = []
                    for index, row in events_df.iterrows():
                        onset = row['onset']
                        duration = row['duration']
                        value = int(row['value'])
                        label = 'Rest' if value == 1 else 'Stimulus' if value == 2 else None
                        if label:
                            n_epochs = int(np.floor(duration / self.epoch_duration))
                            for i in range(n_epochs):
                                epoch_onset = onset + i * self.epoch_duration
                                epoch_sample = int(epoch_onset * self.target_sample_rate)
                                events_sample.append(epoch_sample)
                                event_labels.append(label)

                    # Create events array
                    event_id = {'Rest': 0, 'Stimulus': 1}
                    numeric_labels = [event_id[label] for label in event_labels]
                    events = np.column_stack((events_sample, np.zeros(len(events_sample), dtype=int), numeric_labels))

                    # Create Epochs object
                    tmin = 0
                    tmax = self.epoch_duration - 1 / self.target_sample_rate
                    with self.suppress_output():
                        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, event_repeated='drop')

                    data = epochs.get_data()
                    labels = event_labels  # Use string labels

                    dataset_data.extend(data)
                    dataset_labels.extend(labels)

                # Process 'Rest' task
                task_name = 'Rest'
                eeg_file = os.path.join(eeg_folder, f"{sub_folder}_task-{task_name}_eeg.set")
                if os.path.exists(eeg_file):
                    with self.suppress_output():
                        raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
                        raw.resample(self.target_sample_rate, verbose=False)
                    
                    # Handle channel naming issues
                    raw = self._standardize_channel_names(raw)
                    
                    # Try to set montage, but continue if it fails
                    try:
                        raw = self._match_standard_channels(raw)
                    except Exception as e:
                        print(f"Warning: Could not set standard montage for {sub_folder}. Error: {e}")
                        print("Continuing with existing channels.")

                    # Segment data into fixed-length epochs
                    epochs = self._segment_epochs(raw)
                    data = epochs.get_data()
                    labels = ['Rest'] * len(data)  # Assign label 'Rest'

                    dataset_data.extend(data)
                    dataset_labels.extend(labels)

        print(f"Finished processing {subject_count} subjects for dataset {dataset_id}")
        return dataset_data, dataset_labels

    def _load_ds005420(self, data_dir, dataset_id):
        """
        Load data from the ds005420 dataset.

        Args:
            data_dir (str): Root directory containing the datasets.
            dataset_id (str): Identifier for the ds005420 dataset.

        Returns:
            tuple: (list of EEG epochs, list of corresponding labels)
        """
        dataset_data = []
        dataset_labels = []
        dataset_dir = os.path.join(data_dir, dataset_id + '-download')
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
        subject_count = 0

        for sub_folder in os.listdir(dataset_dir):
            sub_folder_path = os.path.join(dataset_dir, sub_folder)
            if os.path.isdir(sub_folder_path) and sub_folder.startswith('sub-'):
                eeg_folder = os.path.join(sub_folder_path, 'eeg')
                if not os.path.exists(eeg_folder):
                    raise FileNotFoundError(f"EEG folder '{eeg_folder}' does not exist.")
                for task in ['oa', 'oc']:
                    channels_file = os.path.join(eeg_folder, f"{sub_folder}_task-{task}_channels.tsv")
                    eeg_file = os.path.join(eeg_folder, f"{sub_folder}_task-{task}_eeg.edf")
                    if os.path.exists(channels_file) and os.path.exists(eeg_file):
                        # Read the channels.tsv file to get the channel names
                        channels_df = pd.read_csv(channels_file, sep='\t')
                        raw_channel_names = channels_df['name'].tolist()
                        # Process the channel names to extract standard names
                        # For example, 'EEG Fp1-A1A2' -> 'Fp1'
                        channel_names = []
                        for ch_name in raw_channel_names:
                            ch_name = ch_name.replace('EEG ', '')
                            if '-A1A2' in ch_name:
                                ch_name = ch_name.replace('-A1A2', '')
                            elif '-ROC' in ch_name or '-LOC' in ch_name:
                                ch_name = ch_name.replace('-ROC', '').replace('-LOC', '')
                            channel_names.append(ch_name)

                        with self.suppress_output():
                            raw = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
                            # Set the channel names
                            raw.rename_channels({old_name: new_name for old_name, new_name in zip(raw.ch_names, channel_names)})
                            # Set channel types to EEG
                            raw.set_channel_types({ch_name: 'eeg' for ch_name in channel_names})
                            # Standardize channel names and match standard channels
                            raw = self._match_standard_channels(raw)
                            # Resample to target sample rate
                            raw.resample(self.target_sample_rate, verbose=False)
                        # Segment the data into epochs
                        epochs = self._segment_epochs(raw)
                        data = epochs.get_data()
                        # Assign label
                        label = 'OA' if task == 'oa' else 'OC'
                        labels = [label] * len(data)
                        dataset_data.extend(data)
                        dataset_labels.extend(labels)
                subject_count += 1
        print(f"Finished processing {subject_count} subjects for dataset {dataset_id}")
        return dataset_data, dataset_labels
    
    def _standardize_channel_names(self, raw):
        """
        Standardize channel names from 'E' notation to 10-20 system.

        Args:
            raw (mne.io.Raw): Raw EEG data.

        Returns:
            mne.io.Raw: EEG data with standardized channel names.
        """
        channel_mapping = {
            'E1': 'Fp1', 'E2': 'Fp2', 'E3': 'F7', 'E4': 'F3', 'E5': 'Fz', 'E6': 'F4', 
            'E7': 'F8', 'E8': 'T3', 'E9': 'C3', 'E10': 'Cz', 'E11': 'C4', 'E12': 'T4', 
            'E13': 'T5', 'E14': 'P3', 'E15': 'Pz', 'E16': 'P4', 'E17': 'T6', 'E18': 'O1', 
            'E19': 'Oz', 'E20': 'O2', 'E21': 'Fpz', 'E22': 'AF3', 'E23': 'AF4', 'E24': 'F5', 
            'E25': 'F1', 'E26': 'F2', 'E27': 'F6', 'E28': 'FC5', 'E29': 'FC1', 'E30': 'FC2', 
            'E31': 'FC6', 'E32': 'FT10'
        }
        
        # Rename channels that exist in the raw data
        rename_dict = {old: new for old, new in channel_mapping.items() if old in raw.ch_names}
        raw.rename_channels(rename_dict)
        
        return raw

    def _match_standard_channels(self, raw):
        """
        Match the channels in the raw data to the standard channel set.

        Args:
            raw (mne.io.Raw): Raw EEG data.

        Returns:
            mne.io.Raw: Raw EEG data with matched standard channels.
        """
        try:
            with self.suppress_output():
                raw = self._standardize_channel_names(raw)
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.rename_channels({ch_name: ch_name.upper() for ch_name in raw.ch_names})
                raw.set_montage(montage, match_case=False, on_missing='warn')
            available_channels = [ch for ch in self.standard_channels if ch in raw.ch_names]
            if not available_channels:
                print(f"Warning: No standard channels found. Using all available channels: {raw.ch_names}")
                available_channels = raw.ch_names
            raw.pick(available_channels)
        except Exception as e:
            print(f"Warning: Could not set standard montage. Proceeding with existing channels. Error: {e}")
        return raw

    def _segment_epochs(self, raw):
        """
        Segment the raw EEG data into epochs.

        Args:
            raw (mne.io.Raw): Raw EEG data.

        Returns:
            numpy.ndarray: Array of segmented epochs.
        """
        with self.suppress_output():
            raw.filter(1, 40, fir_design='firwin', verbose=False)
            events = mne.make_fixed_length_events(raw, duration=self.epoch_duration, start=0, stop=raw.n_times / raw.info['sfreq'])
            epochs = mne.Epochs(raw, events, tmin=0, tmax=self.epoch_duration, baseline=None, preload=True, verbose=False)
        return epochs

    def plot_example(self, idx):
        """
        Plot the EEG data for a single example.
        
        Args:
            idx (int): Index of the example to plot.
        """
        sample, _ = self[idx]
        
        fig, ax = plt.subplots(figsize=(15, 10))
        for i, channel in enumerate(self.standard_channels):
            ax.plot(sample[i], label=channel)
        
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'EEG Data for Example {idx}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

    def plot_random_examples(self, n=5):
        """
        Plot EEG data for a random selection of examples.
        
        Args:
            n (int): Number of random examples to plot.
        """
        indices = np.random.choice(len(self), n, replace=False)
        for idx in indices:
            self.plot_example(idx)

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (EEG data tensor, label)
        """
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample, dtype=torch.float32), self.labels[idx]

    def standardize_epochs(self, epochs):
        """
        Standardize all epochs to have the same number of channels and time points,
        selecting channels based on importance.

        Args:
            epochs (list): List of EEG epochs.

        Returns:
            list: Standardized list of EEG epochs.
        """
        # Find the minimum number of time points across all epochs
        min_time_points = min(epoch.shape[1] for epoch in epochs)

        # Get all available channels across all epochs
        all_available_channels = set()
        for epoch in epochs:
            all_available_channels.update(self.standard_channels[:epoch.shape[0]])

        # Select the most important available channels
        selected_channels = [ch for ch in self.important_channels if ch in all_available_channels]
        num_channels = len(selected_channels)
        print('initial amount of channels: ', len(self.standard_channels))
        print('initial channel names: ', self.standard_channels)
        print(f"{num_channels} channels in common to all recordings: {selected_channels}")
        print(f"Minimum time points: {min_time_points}")

        standardized_epochs = []
        for epoch in epochs:
            # Create a mapping of channel indices
            channel_mapping = {ch: i for i, ch in enumerate(self.standard_channels[:epoch.shape[0]])}
            
            # Initialize the standardized epoch
            standardized_epoch = np.zeros((num_channels, min_time_points))
            
            for i, channel in enumerate(selected_channels):
                if channel in channel_mapping:
                    channel_idx = channel_mapping[channel]
                    standardized_epoch[i] = epoch[channel_idx, :min_time_points]
            
            standardized_epochs.append(standardized_epoch)

        # Update the standard_channels attribute
        self.standard_channels = selected_channels

        return standardized_epochs

    def _find_bdf_file(self, eeg_folder, sub_folder, session=None):
        """
        Find the .bdf EEG file in the given folder.

        Args:
            eeg_folder (str): Path to the EEG folder.
            sub_folder (str): Name of the subject folder.
            session (str, optional): Session name for Parkinson's disease subjects.

        Returns:
            str: Path to the .bdf EEG file.

        Raises:
            FileNotFoundError: If no matching .bdf file is found.
        """
        for filename in os.listdir(eeg_folder):
            if filename.endswith('_eeg.bdf'):
                if session:
                    expected_prefix = f"{sub_folder}_{session}_task-rest"
                else:
                    expected_prefix = f"{sub_folder}_ses-hc_task-rest"
                if filename.startswith(expected_prefix):
                    return os.path.join(eeg_folder, filename)
        raise FileNotFoundError(f"No .bdf EEG file found in '{eeg_folder}'.")

    def _process_raw(self, raw):
        """
        Process raw EEG data: resample, set montage, and pick standard channels.

        Args:
            raw (mne.io.Raw): Raw EEG data.

        Returns:
            mne.io.Raw: Processed raw EEG data.
        """
        raw.resample(self.target_sample_rate)
        raw = self._match_standard_channels(raw)
        return raw
   
    def create_label_mappings(self):
        """
        Create mappings between string labels and numeric labels.
        """
        unique_labels = sorted(set(self.string_labels))
        label_to_num = {label: i for i, label in enumerate(unique_labels)}
        num_to_label = {i: label for label, i in label_to_num.items()}
        return label_to_num, num_to_label

    def create_class_mapping_csv(self, data_dir):
        """
        Create a CSV file mapping class numbers to class labels.
        
        Args:
            data_dir (str): Root directory to save the CSV file.
        """
        csv_path = os.path.join(data_dir, 'class_mapping.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Class Number', 'Class Label'])
            for class_num, class_label in self.num_to_label.items():
                writer.writerow([class_num, class_label])
        
        print(f"Class mapping CSV created at: {csv_path}")
        
  

    def save_dataset(self, output_dir):
        """
        Save the processed dataset with numeric labels matching the CSV mapping.

        Args:
            output_dir (str): Directory to save the processed dataset.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save data
        np.save(os.path.join(output_dir, 'eeg_data.npy'), np.array(self.data))

        # Save numeric labels
        np.save(os.path.join(output_dir, 'eeg_labels.npy'), np.array(self.labels))

        # Save string labels (for reference)
        np.save(os.path.join(output_dir, 'eeg_string_labels.npy'), np.array(self.string_labels))

        # Save standard channels
        with open(os.path.join(output_dir, 'standard_channels.txt'), 'w') as f:
            for channel in self.standard_channels:
                f.write(f"{channel}\n")

        print(f"Dataset saved in {output_dir}")
        print("Number of samples:", len(self.data))
        print("Label distribution:")
        unique, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label} ({self.num_to_label[label]}): {count}")

        self.create_class_mapping_csv(output_dir)

    def rename_channels(self, rename_dict):
        """
        Rename channels in all epochs.

        Args:
            rename_dict (dict): Dictionary mapping old channel names to new channel names.
        """
        for i in range(len(self.data)):
            for j, channel in enumerate(self.standard_channels):
                if channel in rename_dict:
                    self.standard_channels[j] = rename_dict[channel]

    def set_montage(self, montage):
        """
        Set the montage for the dataset.

        Args:
            montage (mne.channels.DigMontage): The montage to set.
        """
        # This method doesn't actually change the data, but we can use it to update channel information if needed
        self.montage = montage
        print("Montage set for the dataset.")

    def _process_raw_ds002778(self, raw):
        """
        Process raw EEG data for ds002778 dataset: resample and handle channel issues.

        Args:
            raw (mne.io.Raw): Raw EEG data.

        Returns:
            mne.io.Raw: Processed raw EEG data.
        """
        raw.resample(self.target_sample_rate)
        
        raw = self._standardize_channel_names(raw)
        
        # Get the intersection of available channels and standard channels
        available_standard_channels = [ch for ch in self.standard_channels if ch in raw.ch_names]
        
        # If no standard channels are found, use all available channels
        if not available_standard_channels:
            print(f"Warning: No standard channels found. Using all available channels: {raw.ch_names}")
            available_channels = raw.ch_names
        else:
            available_channels = available_standard_channels
        
        # Pick the available channels
        raw.pick(available_channels)
        
        return raw

    @classmethod
    def load_saved_dataset(cls, data_dir):
        """
        Load a previously saved dataset.

        Args:
            data_dir (str): Directory containing the saved dataset files.

        Returns:
            EEGDataset: An instance of EEGDataset with loaded data and labels.
        """
        # Load data and labels
        data = np.load(os.path.join(data_dir, 'eeg_data.npy'), allow_pickle=True)
        labels = np.load(os.path.join(data_dir, 'eeg_labels.npy'), allow_pickle=True)
        string_labels = np.load(os.path.join(data_dir, 'eeg_string_labels.npy'), allow_pickle=True)

        # Create an instance of EEGDataset
        dataset = cls.__new__(cls)
        dataset.data = list(data)
        dataset.labels = list(labels)
        dataset.string_labels = list(string_labels)

        # Load class mapping
        dataset.label_to_num, dataset.num_to_label = dataset._load_class_mapping(data_dir)

        # Set other attributes
        dataset.transform = None
        dataset.target_sample_rate = 250  # Default value, adjust if necessary
        dataset.epoch_duration = 4.0  # Default value, adjust if necessary
        dataset.epoch_samples = int(dataset.target_sample_rate * dataset.epoch_duration)

        # Load standard channels
        dataset.standard_channels = dataset._load_standard_channels(data_dir)

        print(f"Dataset loaded from {data_dir}")
        print("Number of samples:", len(dataset.data))
        print("Label distribution:")
        unique, counts = np.unique(dataset.labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label} ({dataset.num_to_label[label]}): {count}")

        return dataset

    def _load_class_mapping(self, data_dir):
        """
        Load class mapping from CSV file.

        Args:
            data_dir (str): Directory containing the class mapping CSV.

        Returns:
            tuple: (label_to_num, num_to_label) dictionaries.
        """
        csv_path = os.path.join(data_dir, 'class_mapping.csv')
        label_to_num = {}
        num_to_label = {}
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                class_num, class_label = int(row[0]), row[1]
                label_to_num[class_label] = class_num
                num_to_label[class_num] = class_label
        return label_to_num, num_to_label

    def _load_standard_channels(self, data_dir):
        """
        Load standard channels from a text file.

        Args:
            data_dir (str): Directory containing the standard channels file.

        Returns:
            list: List of standard channel names.
        """
        channels_path = os.path.join(data_dir, 'standard_channels.txt')
        if os.path.exists(channels_path):
            with open(channels_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            print("Warning: standard_channels.txt not found. Using default channels.")
            return self.standard_channels

    def save_dataset_examples(self, dataset_id, plots_dir):
        """
        Save plots of examples from the specified dataset, one for each class.

        Args:
            dataset_id (str): The ID of the dataset to plot examples from.
            plots_dir (str): The directory to save the plots in.
        """
        # Create a folder for the dataset if it doesn't exist
        dataset_plots_dir = os.path.join(plots_dir, dataset_id)
        os.makedirs(dataset_plots_dir, exist_ok=True)

        # Filter indices for the current dataset
        dataset_indices = [i for i, label in enumerate(self.string_labels) if dataset_id in label]

        if not dataset_indices:
            print(f"No examples found for dataset {dataset_id}")
            return

        # Get unique classes for this dataset
        dataset_classes = set(self.string_labels[i] for i in dataset_indices)

        for class_label in dataset_classes:
            # Find an example of this class
            example_index = next(i for i in dataset_indices if self.string_labels[i] == class_label)
            data, label = self[example_index]
            data_numpy = data.numpy()

            # Apply filtering (assuming 1-40 Hz bandpass filter)
            filtered_data = self._apply_filter(data_numpy)

            # Determine overall min and max for consistent y-axis
            overall_min = min(data_numpy.min(), filtered_data.min())
            overall_max = max(data_numpy.max(), filtered_data.max())

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 30))

            # Plot original data
            self._plot_data(data_numpy, class_label, ax1, "Original", overall_min, overall_max)

            # Plot filtered data
            self._plot_data(filtered_data, class_label, ax2, "Filtered (1-40 Hz)", overall_min, overall_max)

            plt.tight_layout()
            plt.savefig(os.path.join(dataset_plots_dir, f'{class_label.replace("/", "_")}_example.png'))
            plt.close()

        print(f"Plots for dataset {dataset_id} saved in {dataset_plots_dir}")

    def _plot_data(self, data, class_label, ax, title_prefix, y_min, y_max):
        """Helper method to plot data for all electrodes."""
        time = np.linspace(0, self.epoch_duration, data.shape[1])
        for i, electrode in enumerate(self.standard_channels):
            ax.plot(time, data[i], label=f'{electrode}')

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title(f'{title_prefix} EEG Data - {class_label}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        ax.set_ylim(y_min, y_max)  # Set consistent y-axis limits

    def _apply_filter(self, data):
        """Apply a 1-40 Hz bandpass filter to the data."""
        nyquist = 0.5 * self.target_sample_rate
        low = 1 / nyquist
        high = 40 / nyquist
        b, a = butter(4, [low, high], btype='band')
        
        return filtfilt(b, a, data)


