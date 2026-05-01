import os
import re
import pickle

from models.nnsr import Discriminant

import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning as L

def load_results(output_dir):

    nnsr_dir = os.path.join(output_dir, 'nnsr')
    logs_dir = os.path.join(nnsr_dir, 'lightning_logs')

    with open(os.path.join(nnsr_dir, 'events_signal_train.pkl'), 'rb') as f:
        events_sig_train = pickle.load(f)
    with open(os.path.join(nnsr_dir, 'events_background_train.pkl'), 'rb') as f:
        events_bkg_train = pickle.load(f)
    with open(os.path.join(nnsr_dir, 'events_signal_val.pkl'), 'rb') as f:
        events_sig_val = pickle.load(f)
    with open(os.path.join(nnsr_dir, 'events_background_val.pkl'), 'rb') as f:
        events_bkg_val = pickle.load(f)
    with open(os.path.join(nnsr_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # Find the latest version folder
    versions = [d for d in os.listdir(logs_dir) if re.match(r'version_\d+', d)]
    if not versions:
        raise FileNotFoundError("No version folders found in lightning_logs.")

    # Extract version numbers and sort
    latest_version = max(versions, key=lambda v: int(re.search(r'\d+', v).group()))
    checkpoint_dir = os.path.join(logs_dir, latest_version, 'checkpoints')

    # Find all checkpoint files matching the pattern
    checkpoints = [f for f in os.listdir(checkpoint_dir) if re.match(r'epoch=\d+-val_loss=.+\.ckpt', f)]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Get the checkpoint with the largest epoch number
    ckpt_path = max(checkpoints, key=lambda f: int(re.search(r'epoch=(\d+)', f).group(1)))

    ckpt = Discriminant.load_from_checkpoint(checkpoint_path=os.path.join(checkpoint_dir, ckpt_path))

    return (events_sig_train, events_sig_val), (events_bkg_train, events_bkg_val), scaler, ckpt

def get_nnsr_decision(events, features, scaler_X, model):
    trainer = L.Trainer(accelerator='gpu', devices=1)

    kinematics = events.kinematics[features]
    X_nnsr = scaler_X.transform(kinematics.to_numpy())
    dl_nnsr = DataLoader(TensorDataset(torch.tensor(X_nnsr, dtype=torch.float32)), batch_size=1024, num_workers=1) 
    nnsr_dec = torch.cat(trainer.predict(model, dl_nnsr))

    return nnsr_dec.numpy().flatten()