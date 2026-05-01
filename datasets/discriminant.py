import os
import pickle
import re

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

from physics.simulation import mcfm

class DiscriminantDataModule(L.LightningDataModule):

    def __init__(self, background_events: str = '', signal_events: str = '', analysis = 'h4l', features = ['cth_star', 'cth_1', 'cth_2', 'phi_1', 'phi', 'Z1_mass', 'Z2_mass', '4l_mass', '4l_rapidity'], sample_size = 10000, batch_size: int = 32, random_state: int=None, data_dir : str = './'):
        super().__init__()

        self.features = features

        self.background_file = background_events
        self.signal_file = signal_events

        self.sample_size = sample_size

        self.batch_size = batch_size
        self.random_state = random_state

        self.data_dir = data_dir
        self.scaler = StandardScaler()

    def prepare_data(self):

        events_background = mcfm.from_csv(cross_section=None, file_path=self.background_file, kinematics=self.features)
        events_signal = mcfm.from_csv(cross_section=None, file_path=self.signal_file, kinematics=self.features)

        events_background = events_background.sample(self.sample_size, random_state=self.random_state)
        events_signal = events_signal.sample(self.sample_size, random_state=self.random_state)

        train_size, val_size, test_size = 6, 2, 2
        events_background_train, events_background_val, events_background_test = events_background.split(train_size=train_size, val_size=val_size, test_size=test_size)
        events_signal_train, events_signal_val, events_signal_test = events_signal.split(train_size=train_size, val_size=val_size, test_size=test_size)

        self.training_data = DiscriminantDataset(events_background_train, events_signal_train, self.features, scaler = None, random_state = self.random_state)
        self.scaler.fit(self.training_data.X)

        # save stuff for later
        with open(os.path.join(self.data_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(self.data_dir, 'events_background_train.pkl'), 'wb') as f:
            pickle.dump(events_background_train, f)
        with open(os.path.join(self.data_dir, 'events_signal_train.pkl'), 'wb') as f:
            pickle.dump(events_signal_train, f)
        with open(os.path.join(self.data_dir, 'events_background_val.pkl'), 'wb') as f:
            pickle.dump(events_background_val, f)
        with open(os.path.join(self.data_dir, 'events_signal_val.pkl'), 'wb') as f:
            pickle.dump(events_signal_val, f)
        with open(os.path.join(self.data_dir, 'events_background_test.pkl'), 'wb') as f:
            pickle.dump(events_background_test, f)
        with open(os.path.join(self.data_dir, 'events_signal_test.pkl'), 'wb') as f:
            pickle.dump(events_signal_test, f)

    def setup(self, stage: str):

        if stage == 'fit':

            with open(os.path.join(self.data_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_background_train.pkl'), 'rb') as f:
                events_background_train = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_signal_train.pkl'), 'rb') as f:
                events_signal_train = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_background_val.pkl'), 'rb') as f:
                events_background_val = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_signal_val.pkl'), 'rb') as f:
                events_signal_val = pickle.load(f)

            self.training_data = DiscriminantDataset(events_background_train, events_signal_train, self.features, scaler = self.scaler, random_state=self.random_state)
            self.validation_data = DiscriminantDataset(events_background_val, events_signal_val, self.features, scaler = self.scaler, random_state=self.random_state)

        elif stage == 'test':
            with open(os.path.join(self.data_dir, 'events_background_test.pkl'), 'rb') as f:
                events_background_test = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_signal_test.pkl'), 'rb') as f:
                events_signal_test = pickle.load(f)

            self.testing_data = DiscriminantDataset(events_background_test, events_signal_test, self.features, scaler = self.scaler, random_state=self.random_state)

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.testing_data, batch_size=self.batch_size, num_workers=8)

class DiscriminantDataset(Dataset):
    def __init__(self, events_background = None, events_signal = None, features = None, scaler = None, random_state = None):
        super().__init__()

        # get features
        X_background = events_background.kinematics[features].to_numpy()
        X_signal = events_signal.kinematics[features].to_numpy()
        self.X = np.concatenate([X_background, X_signal])

        # weights
        w_background = events_background.weights.to_numpy()
        w_signal = events_signal.weights.to_numpy()
        self.w = np.concatenate([w_background, w_signal])

        # signal = 1, background = 0
        self.s = np.concatenate([np.zeros_like(w_background), np.ones_like(w_signal)])

        if scaler is not None:
            self.X = scaler.transform(self.X)
        
        self.X, self.s, self.w = shuffle(self.X, self.s, self.w, random_state=random_state)
    
    def __len__(self):
        return len(self.s)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.s[index], dtype=torch.float32), torch.tensor(self.w[index], dtype=torch.float32)
