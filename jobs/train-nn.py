import sys
sys.path.append('../')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Nadam 

from hstar import gghzz, c6, msq

from hzz import zpair, angles

from nn import datasets, models

import numpy as np

import os, json

from sklearn.preprocessing import StandardScaler

from argparse import ArgumentParser

SEED=373485

def parse_arguments():
    parser = ArgumentParser(prog='',
                            description='',
                            epilog='')
    parser.add_argument('-l', '--learn-rate', help='Learning rate used with the Nadam optimizer. (default: 1e-5)')
    parser.add_argument('-b', '--batch-size', help='Batch size for training the NN. (default: 32)')
    parser.add_argument('-e', '--epochs', help='Maximum number of epochs in training. (default: 100)')
    parser.add_argument('-n', '--num-events', help='Number of events N per class and c6 value. (default: full dataset)')
    parser.add_argument('--num-layers', help='Number of dense layers between input and output layers in the NN. (default: 10)')
    parser.add_argument('--num-nodes', help='Number of nodes per layer or comma separated numbers of nodes for all dense layers. (default: 2000)')
    parser.add_argument('-s', '--sample-dir', help='Path to the directory containing the sample files. (default: ../../)')
    parser.add_argument('-o', '--output-dir', help='Path to the directory where outputs files will be placed. Will be created is it does not exist yet. (default: ./)')
    parser.add_argument('--c6', help='Single value for c6 or three comma separated values like a,b,c specifying a np.linspace(a,b,c). (default: -10,10,21)')
    parser.add_argument('--sig', action='store_true', help='Use for enabling training on SIG data only.')
    parser.add_argument('--int', action='store_true', help='Use for enabling training on INT data only.')

    args = parser.parse_args()

    sample_dir = '../..' if args.sample_dir is None else str(args.sample_dir)
    output_dir = '.' if args.output_dir is None else str(args.output_dir)
    learn_rate = 1e-5 if args.learn_rate is None else float(args.learn_rate)
    batch_size = 32 if args.batch_size is None else int(args.batch_size)
    num_events = None if args.num_events is None else int(args.num_events)
    num_layers = 10 if args.num_layers is None else int(args.num_layers)
    epochs = 100 if args.epochs is None else int(args.epochs)
    train_sig = False if args.sig is None else True
    train_int = False if args.int is None else True

    if train_sig and train_int:
        raise ValueError('--int and --sig cannot be enabled simultaneously')

    c6_input = np.array(-10,10,21) if args.c6 is None else np.fromstring(args.c6, sep=',')

    if len(c6_input) == 1:
        c6_values = c6_input
    elif len(c6_input) == 3:
        c6_values = np.linspace(float(c6_input[0]), float(c6_input[1]), int(c6_input[2]))
    else:
        raise ValueError('c6 should be a single value or three comma separated values like a,b,c specifying a np.linspace(a,b,c)')
    
    n_nodes_input = np.array([2000]) if args.num_nodes is None else np.fromstring(args.num_nodes.replace('[','').replace(']',''), sep=',')

    if len(n_nodes_input) == 1:
        num_nodes = n_nodes_input.item()
    elif len(n_nodes_input) == num_layers:
        num_nodes = n_nodes_input.tolist()
    else:
        raise ValueError('num_nodes should be a single value or a comma separated list of integer values fulfilling len(num_nodes)=num_layers')
    

    return {'sample_dir': sample_dir, 'output_dir': output_dir, 'sig': train_sig, 'int': train_int, 'learning_rate': learn_rate, 'batch_size': batch_size, 'num_events': num_events, 'num_layers': num_layers, 'num_nodes': num_nodes, 'epochs': epochs, 'c6_values': c6_values.tolist()}


def load_kinematics(sample, bounds1=(50,115), bounds2=(50,115), algorithm='leastsquare'):
    z_chooser = zpair.ZPairChooser(bounds1=bounds1, bounds2=bounds2, algorithm=algorithm)

    return angles.calculate(*sample.events.filter(z_chooser))


def save_config(output_dir, *config):
    file_path = os.path.join(output_dir, 'job.config')

    os.makedirs(output_dir, exist_ok=True)
    with open(file_path, 'w') as config_file:
        config_file.write(json.dumps(config, indent=4))


def main():
    config = parse_arguments()

    if config['num_events'] is None:
        n_i = None
    else:
        n_i = int(2*config['num_events']/3)

    mirrored_strategy = tf.distribute.MirroredStrategy()

    sample = gghzz.Process(  
        (1.4783394, os.path.join(config['sample_dir'], 'ggZZ2e2m_all_new.csv'), n_i),
        (0.47412769, os.path.join(config['sample_dir'], 'ggZZ4e_all_new.csv'), n_i),
        (0.47412769, os.path.join(config['sample_dir'], 'ggZZ4m_all_new.csv'), n_i)
    )

    set_size = sample.events.kinematics.shape[0]

    kin_variables = load_kinematics(sample)

    true_size = kin_variables.shape[0]

    print(f'Initial base size set to {int(set_size/2)}. Train and validation data will be {int(true_size/2)*2} each after Z mass cuts.')

    component = msq.Component.SIG if config['sig'] else msq.Component.SBI
    component = msq.Component.INT if config['int'] else component

    c6_mod = c6.Modifier(amplitude_component = component, c6_values = [-5,-1,0,1,5])
    c6_weights, c6_prob = c6_mod.modify(sample=sample, c6=config['c6_values'])

    train_data = datasets.build_dataset_tf(x_arr = kin_variables[:int(true_size/2)], 
                                           param_values = config['c6_values'], 
                                           signal_weights = c6_weights[:int(true_size/2)], 
                                           background_weights = np.array(sample[component].weights)[:int(true_size/2)], 
                                           normalization = 1)

    val_data = datasets.build_dataset_tf(x_arr = kin_variables[int(true_size/2):], 
                                         param_values = config['c6_values'], 
                                         signal_weights = c6_weights[int(true_size/2):], 
                                         background_weights = np.array(sample[component].weights)[int(true_size/2):], 
                                         normalization = 1)
    
    # The following will scale only kinematics for nonprm and kinematics + c6 for prm
    train_scaler = StandardScaler()
    train_data = tf.concat([train_scaler.fit_transform(train_data[:,:-2]), train_data[:,-2:]], axis=1)
    train_data = tf.random.shuffle(train_data, seed=SEED)

    val_data = tf.concat([train_scaler.transform(val_data[:,:-2]), val_data[:,-2:]], axis=1)
    val_data = tf.random.shuffle(val_data, seed=SEED)

    save_config(config['output_dir'], config, {'scaler.mean_': train_scaler.mean_.tolist(), 'scaler.var_': train_scaler.var_.tolist(), 'scaler.scale_': train_scaler.scale_.tolist()})
    print(f'Settings for this run are stored in {os.path.join(config["output_dir"], "job.config")}')

    with mirrored_strategy.scope():
        
        if len(config['c6_values']) == 1:
            model = models.C6_4l_clf(num_layers=config['num_layers'], num_nodes=config['num_nodes'], input_dim=kin_variables.shape[1])
        else:
            model = models.C6_4l_clf(num_layers=config['num_layers'], num_nodes=config['num_nodes'], input_dim=kin_variables.shape[1]+1)

        optimizer = Nadam(
            learning_rate=config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'], weighted_metrics=['binary_accuracy'])

    os.makedirs(config['output_dir'], exist_ok=True)

    checkpoint_filepath = os.path.join(config['output_dir'], 'checkpoint.model.tf')
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, save_format='tf')
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=30, start_from_epoch=30)

    history_callback = model.fit(x=train_data[:,:-2], y=train_data[:,-2][:,np.newaxis], sample_weight=train_data[:,-1][:,np.newaxis], validation_data=(val_data[:,:-2], val_data[:,-2], val_data[:,-1]), batch_size=config['batch_size'], callbacks=[model_checkpoint_callback, early_stopping_callback], epochs=config['epochs'], verbose=2)

    model.save(os.path.join(config['output_dir'], 'final.model.tf'), save_format='tf')

    with open(os.path.join(config['output_dir'], 'history.txt'), 'w') as hist_file:
        hist_file.write(str(history_callback.history['loss']))
        hist_file.write(str(history_callback.history['val_loss']))

    print(model.summary())


if __name__ == '__main__':
    main()