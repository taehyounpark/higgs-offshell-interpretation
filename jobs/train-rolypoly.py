import numpy as np
import os, json
import sys
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import tensorflow as tf

from nn.rolypoly import dataset, model


SEED=373485

def parse_arguments():
    parser = ArgumentParser(description='Python script for training (deep) neural networks in a SM vs BSM classification scenario.')
    parser.add_argument('config', type=str, help='Config file to be used for setting necessary parameters.')

    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Only one of the component flags can be activated at once
    component_flags = np.array(['sig', 'sbi', 'int', 'bkg'])
    num_c_flags = np.sum(np.array([c_flag in config['flags'] for c_flag in component_flags]).astype(int))

    if num_c_flags > 1:
        raise ValueError(f'You can only activate one of {component_flags} at once')

    # Add all active flags to flag list
    flags_active = []
    flags_possible = ['distributed']
    flags_possible.extend(component_flags)
    for flag in flags_possible:
        if flag in config['flags']:
            flags_active.append(flag)
    
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

    # Check if coefficient index is possible value
    coeff = int(config['coefficient'])
    if coeff not in range(1,5):
        raise ValueError('The coefficient index has to be an integer value in {1,2,3,4}')

    return {'sample_dir': sample_dir, 'coeff': coeff, 'flags': flags_active, 'learning_rate': config['learning_rate'], 'batch_size': config['batch_size'], 'num_events': config['num_events'], 'num_layers': config['num_layers'], 'num_nodes': num_nodes, 'epochs': config['epochs']}


def main(config):
    rng = np.random.default_rng(seed=SEED)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    if 'distributed' in config['flags']:
        print(f'Model will be training on {mirrored_strategy.num_replicas_in_sync} GPUs')

    # Build datasets (distributed if flag given)
    train_dataset, val_dataset = dataset.build(config, SEED, mirrored_strategy)

    # Build model (distributed if flag given)
    model_rolypoly = model.build(config, mirrored_strategy)

    # Train model
    history_callback = model.train(model_rolypoly, config, train_dataset, val_dataset, strategy=mirrored_strategy)
    
    # Save model
    model.save(model_rolypoly, history_callback)

    print(model_rolypoly.summary())


if __name__ == '__main__':
    args = parse_arguments()
    config = load_config(args.config)

    main(config)
