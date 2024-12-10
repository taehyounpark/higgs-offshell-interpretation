import numpy as np
import os, json
import sys
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import tensorflow as tf

from nn.carl import dataset, model


SEED=373485

def parse_arguments():
    parser = ArgumentParser(description='Python script for training (deep) neural networks in a SM vs BSM classification scenario.')
    parser.add_argument('config', type=str, help='Config file to be used for setting necessary parameters.')

    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.loads(''.join(config_file.readlines()))

    # Only one of the component flags can be activated at once
    component_flags = np.array(['sig-vs-sig', 'sig-vs-sbi', 'int-vs-sbi', 'bkg-vs-sbi'])
    num_c_flags = np.sum(np.array([c_flag in config['flags'] for c_flag in component_flags]).astype(int))

    if num_c_flags > 1:
        raise ValueError('You can only activate one of [sig-vs-sig, sig-vs-sbi, int-vs-sbi, bkg-vs-sbi] at once')

    # Add all active flags to flag list
    flags_active = []
    flags_possible = ['distributed', 'sig-vs-sig', 'sig-vs-sbi', 'int-vs-sbi', 'bkg-vs-sbi']
    for flag in flags_possible:
        if flag in config['flags']:
            flags_active.append(flag)

    # For BKG no c6 is needed
    if 'bkg-vs-sbi' in flags_active:
        c6_input = np.array([0.0])
    else:
        c6_input = np.fromstring(config['c6_values'].replace('[','').replace(']',''), sep=',')

    # Build c6 array from the input to the c6 argument
    if len(c6_input) == 1:
        c6_values = c6_input
    elif len(c6_input) == 3:
        c6_values = np.linspace(float(c6_input[0]), float(c6_input[1]), int(c6_input[2]))
    else:
        raise ValueError('c6 should be a single value or three comma separated values like a,b,c specifying a np.linspace(a,b,c)')
    
    # Build num_nodes array from the input to the num-nodes argument
    n_nodes_input = np.fromstring(str(config['num_nodes']).replace('[','').replace(']',''), sep=',').astype(int)
    if len(n_nodes_input) == 1:
        num_nodes = n_nodes_input.item()
    elif len(n_nodes_input) == config['num_layers']:
        num_nodes = n_nodes_input.tolist()
    else:
        raise ValueError('num-nodes should be a single value or a comma separated list of integer values with a length equals len(num-nodes)=num-layers')
    
    # Load sample dir from config
    sample_dir = '/'.join([os.environ[el[1:]] if '$' in el else el for el in config['sample_dir'].split('/')])

    return {'sample_dir': sample_dir, 'flags': flags_active, 'learning_rate': config['learning_rate'], 'batch_size': config['batch_size'], 'num_events': config['num_events'], 'num_layers': config['num_layers'], 'num_nodes': num_nodes, 'epochs': config['epochs'], 'c6_values': c6_values.tolist()}


def main(config):
    rng = np.random.default_rng(seed=SEED)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    if 'distributed' in config['flags']:
        print(f'Model will be training on {mirrored_strategy.num_replicas_in_sync} GPUs')

    # Build datasets (distributed if flag given)
    train_dataset, val_dataset = dataset.build(config, SEED, mirrored_strategy)

    # Build model (distributed if flag given)
    model_carl = model.build(config, mirrored_strategy)

    # Train model
    history_callback = model.train(model_carl, config, train_dataset, val_dataset, strategy=mirrored_strategy)
    
    # Save model
    model.save(model_carl, history_callback)

    print(model_carl.summary())


if __name__ == '__main__':
    args = parse_arguments()
    config = load_config(args.config)

    main(config)