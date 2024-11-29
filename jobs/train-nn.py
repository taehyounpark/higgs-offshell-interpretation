import numpy as np
import os, json
import sys
sys.path.append('../')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Nadam 

from physics.hstar import gghzz, c6
from physics.simulation import msq

from physics.hzz import zpair, angles

from nn import datasets, models

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
    parser.add_argument('--sig-vs-sbi', action='store_true', help='Use for enabling training on SIG(c6) vs SBI(SM).')
    parser.add_argument('--int-vs-sbi', action='store_true', help='Use for enabling training on INT(c6) vs SBI(SM).')
    parser.add_argument('--bkg-vs-sbi', action='store_true', help='Use for enabling training on BKG(c6) vs SBI(SM).')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed learning (experimental).')

    args = parser.parse_args()

    sample_dir = '../..' if args.sample_dir is None else str(args.sample_dir)
    output_dir = '.' if args.output_dir is None else str(args.output_dir)
    learn_rate = 1e-5 if args.learn_rate is None else float(args.learn_rate)
    batch_size = 32 if args.batch_size is None else int(args.batch_size)
    num_events = None if args.num_events is None else int(args.num_events)
    num_layers = 10 if args.num_layers is None else int(args.num_layers)
    epochs = 100 if args.epochs is None else int(args.epochs)

    num_flags_components = np.sum(np.array([args.sig, args.int, args.sig_vs_sbi, args.int_vs_sbi, args.bkg_vs_sbi]).astype(int))

    if num_flags_components > 1:
        raise ValueError('You can only activate one of [sig, int, sig-vs-sbi, int-vs-sbi, bkg-vs-sbi] at once')

    flag_component = np.array(['sig', 'int', 'sig-vs-sbi', 'int-vs-sbi', 'bkg-vs-sbi'])[np.where(np.array([args.sig, args.int, args.sig_vs_sbi, args.int_vs_sbi, args.bkg_vs_sbi])==True)]
    flag_component = flag_component[0] if flag_component.shape[0] != 0 else ''

    flags_values = {'distributed': args.distributed}
    flags_active = [flag_component]

    for key,value in flags_values.items():
        if value is True:
            flags_active.append(key)

    if 'bkg-vs-sbi' in flags_active:
        c6_input = np.array([0.0])
    else:
        c6_input = np.array([-10,10,21]) if args.c6 is None else np.fromstring(args.c6, sep=',')

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
    

    return {'sample_dir': sample_dir, 'output_dir': output_dir, 'flags': flags_active, 'learning_rate': learn_rate, 'batch_size': batch_size, 'num_events': num_events, 'num_layers': num_layers, 'num_nodes': num_nodes, 'epochs': epochs, 'c6_values': c6_values.tolist()}


def load_kinematics(sample, bounds1=(50,115), bounds2=(50,115), algorithm='leastsquare'):
    z_chooser = zpair.ZPairChooser(bounds1=bounds1, bounds2=bounds2, algorithm=algorithm)

    return angles.calculate(*sample.events.filter(z_chooser))


def save_config(output_dir, *config):
    file_path = os.path.join(output_dir, 'job.config')

    os.makedirs(output_dir, exist_ok=True)
    with open(file_path, 'w') as config_file:
        config_file.write(json.dumps(config, indent=4))

def get_components(config):
    component_flag = np.array(config['flags'])[np.where(np.array(config['flags']) in ['sig', 'int', 'sig-vs-sbi', 'int-vs-sbi', 'bkg-vs-sbi'])]
    component_flag = component_flag[0] if component_flag.shape[0] != 0 else 'sbi'
    component_1, component_2 = component_flag.split('-')[0], component_flag.split('-')[-1]
    
    comp_dict = {'sig': msq.Component.SIG,
                 'int': msq.Component.INT,
                 'bkg': msq.Component.BKG,
                 'sbi': msq.Component.SBI}

    return (comp_dict[component_1], comp_dict[component_2])
    

def build_model(config, strategy=None):
    if 'distributed' in config['flags'] and strategy is not None:
        with strategy.scope():
            if len(config['c6_values']) == 1:
                model = models.C6_4l_clf(num_layers=config['num_layers'], num_nodes=config['num_nodes'], input_dim=8)
            else:
                model = models.C6_4l_clf(num_layers=config['num_layers'], num_nodes=config['num_nodes'], input_dim=9)

            optimizer = Nadam(
                learning_rate=config['learning_rate'],
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )

            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'], weighted_metrics=['binary_accuracy'])
    else:
        if len(config['c6_values']) == 1:
            model = models.C6_4l_clf(num_layers=config['num_layers'], num_nodes=config['num_nodes'], input_dim=8)
        else:
            model = models.C6_4l_clf(num_layers=config['num_layers'], num_nodes=config['num_nodes'], input_dim=9)

        optimizer = Nadam(
            learning_rate=config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'], weighted_metrics=['binary_accuracy'])
    
    return model

def train_model(model, config, training_data, validation_data, callbacks=None, strategy=None):
    train_dataset = tf.data.Dataset.from_tensor_slices((training_data[:,:-2], training_data[:,-2][:,tf.newaxis], training_data[:,-1][:,tf.newaxis]))
    val_dataset = tf.data.Dataset.from_tensor_slices((validation_data[:,:-2], validation_data[:,-2][:,tf.newaxis], validation_data[:,-1][:,tf.newaxis]))

    if 'distributed' in config['flags'] and strategy is not None:
        train_dataset = train_dataset.batch(config['batch_size']*strategy.num_replicas_in_sync)
        val_dataset = val_dataset.batch(config['batch_size']*strategy.num_replicas_in_sync)

        dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        dist_val_dataset = strategy.experimental_distribute_dataset(val_dataset)

        # Run model.fit
        train_steps = int(training_data.shape[0]/config['batch_size']/strategy.num_replicas_in_sync)
        val_steps = int(validation_data.shape[0]/config['batch_size']/strategy.num_replicas_in_sync)

        history_callback = model.fit(dist_train_dataset, steps_per_epoch=train_steps, validation_data=dist_val_dataset, validation_steps=val_steps, callbacks=callbacks, epochs=config['epochs'], verbose=2)
    else:
        train_dataset = train_dataset.batch(config['batch_size'])
        val_dataset = val_dataset.batch(config['batch_size'])

        # Run model.fit
        history_callback = model.fit(train_dataset, validation_data=val_dataset, callbacks=callbacks, epochs=config['epochs'], verbose=2)
        
    return history_callback


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

    sample.events.filter(msq.ZeroMSQFilter('msq_int_sm'))

    kin_variables = load_kinematics(sample)

    true_size = kin_variables.shape[0]

    print(f'Initial base size set to {int(set_size/2)}. Train and validation data will be {int(true_size/2)*2} each after Z mass cuts.')


    component_c6, component_sm = get_components(config)

    if component_c6 != msq.Component.BKG:
        c6_mod = c6.Modifier(amplitude_component = component_c6, c6_values = [-5,-1,0,1,5])
        c6_weights, c6_prob = c6_mod.modify(sample=sample, c6=config['c6_values'])

        train_data = datasets.build_dataset_tf(x_arr = kin_variables[:int(true_size/2)], 
                                           param_values = config['c6_values'], 
                                           signal_weights = c6_weights[:int(true_size/2)], 
                                           background_weights = np.array(sample[component_sm].weights)[:int(true_size/2)], 
                                           normalization = 1)

        val_data = datasets.build_dataset_tf(x_arr = kin_variables[int(true_size/2):], 
                                            param_values = config['c6_values'], 
                                            signal_weights = c6_weights[int(true_size/2):], 
                                            background_weights = np.array(sample[component_sm].weights)[int(true_size/2):], 
                                            normalization = 1)
    else:
        bkg_weights, bkg_prob = np.array(sample[component_c6].weights)[:,np.newaxis], np.array(sample[component_c6].probabilities)[:,np.newaxis]

        train_data = datasets.build_dataset_tf(x_arr = kin_variables[:int(true_size/2)], 
                                               param_values = [0.0], 
                                               signal_weights = bkg_weights[:int(true_size/2)], 
                                               background_weights = np.array(sample[component_sm].weights)[:int(true_size/2)], 
                                               normalization = 1)

        val_data = datasets.build_dataset_tf(x_arr = kin_variables[int(true_size/2):], 
                                             param_values = [0.0], 
                                             signal_weights = bkg_weights[int(true_size/2):], 
                                             background_weights = np.array(sample[component_sm].weights)[int(true_size/2):], 
                                             normalization = 1)

    
    # The following will scale only kinematics for nonprm and kinematics + c6 for prm
    train_scaler = StandardScaler()
    train_data = tf.concat([train_scaler.fit_transform(train_data[:,:-2]), train_data[:,-2:]], axis=1)
    train_data = tf.random.shuffle(train_data, seed=SEED)

    val_data = tf.concat([train_scaler.transform(val_data[:,:-2]), val_data[:,-2:]], axis=1)
    val_data = tf.random.shuffle(val_data, seed=SEED)

    save_config(config['output_dir'], config, {'scaler.mean_': train_scaler.mean_.tolist(), 'scaler.var_': train_scaler.var_.tolist(), 'scaler.scale_': train_scaler.scale_.tolist()})
    print(f'Settings for this run are stored in {os.path.join(config["output_dir"], "job.config")}')

    # Build model (distributed if flag given)
    model = build_model(config, mirrored_strategy)

    os.makedirs(config['output_dir'], exist_ok=True)

    # Setup keras callbacks
    checkpoint_filepath = os.path.join(config['output_dir'], 'checkpoint.model.tf')
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, save_format='tf')
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=30, start_from_epoch=30)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(config['output_dir'], 'logs'))

    # Train model
    history_callback = train_model(model, config, train_data, val_data, callbacks=[model_checkpoint_callback, tensorboard_callback], strategy=mirrored_strategy)

    model.save(os.path.join(config['output_dir'], 'final.model.tf'), save_format='tf')

    with open(os.path.join(config['output_dir'], 'history.txt'), 'w') as hist_file:
        hist_file.write(str(history_callback.history['loss']))
        hist_file.write(str(history_callback.history['val_loss']))

    print(model.summary())


if __name__ == '__main__':
    main()