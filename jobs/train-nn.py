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

def load_samples(config, component_1, component_2):
    if config['num_events'] is None:
        n_i = None
    else:
        n_i = int(config['num_events']*1.2)#int(2*config['num_events']/3)

    def match_comp(config, component, n_i):
        match component:
            case msq.Component.SIG:
                sample = gghzz.Process(msq.Component.SIG, (0.1, os.path.join(config['sample_dir'], 'ggZZ2e2m_sig.csv'), n_i))
            case msq.Component.SBI:
                sample = gghzz.Process(msq.Component.SBI, (1.5, os.path.join(config['sample_dir'], 'ggZZ2e2m_sbi.csv'), n_i))
            case msq.Component.BKG:
                sample = gghzz.Process(msq.Component.BKG, (1.6, os.path.join(config['sample_dir'], 'ggZZ2e2m_bkg.csv'), n_i))
            case msq.Component.INT:
                sample = gghzz.Process(msq.Component.INT, (-0.2, os.path.join(config['sample_dir'], 'ggZZ2e2m_int.csv'), n_i))
        return sample

    return (match_comp(config, component_1, n_i), match_comp(config, component_2, n_i))



def load_kinematics(sample, bounds1=(50,115), bounds2=(50,115), algorithm='leastsquare'):
    z_chooser = zpair.ZPairChooser(bounds1=bounds1, bounds2=bounds2, algorithm=algorithm)

    return angles.calculate(*sample.events.filter(z_chooser))


def save_config(output_dir, *config):
    file_path = os.path.join(output_dir, 'job.config')

    os.makedirs(output_dir, exist_ok=True)
    with open(file_path, 'w') as config_file:
        config_file.write(json.dumps(config, indent=4))

def get_components(config):
    component_flag = np.array(config['flags'])[np.where([ (flag in ['sig', 'int', 'sig-vs-sbi', 'int-vs-sbi', 'bkg-vs-sbi']) for flag in config['flags'] ])]
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
        with strategy.scope():
            train_dataset = train_dataset.batch(config['batch_size']*strategy.num_replicas_in_sync)
            val_dataset = val_dataset.batch(config['batch_size']*strategy.num_replicas_in_sync)

            history_callback = model.fit(train_dataset, validation_data=val_dataset, callbacks=callbacks, epochs=config['epochs'], verbose=2)
    else:
        train_dataset = train_dataset.batch(config['batch_size'])
        val_dataset = val_dataset.batch(config['batch_size'])

        # Run model.fit
        history_callback = model.fit(train_dataset, validation_data=val_dataset, callbacks=callbacks, epochs=config['epochs'], verbose=2)
        
    return history_callback


def main():
    config = parse_arguments()

    mirrored_strategy = tf.distribute.MirroredStrategy()
    if 'distributed' in config['flags']:
        print(f'Model will be training on {mirrored_strategy.num_replicas_in_sync} GPUs')

    component_1, component_2 = get_components(config)

    sample_1, sample_2 = load_samples(config, component_1, component_2)

    set_size_1, set_size_2 = sample_1.events.kinematics.shape[0], sample_2.events.kinematics.shape[0]

    sample_1.events.filter(msq.MSQFilter('msq_bkg_sm', value=0.0))
    sample_1.events.filter(msq.MSQFilter('msq_bkg_sm', value=np.nan))
    sample_2.events.filter(msq.MSQFilter('msq_bkg_sm', value=0.0))
    sample_2.events.filter(msq.MSQFilter('msq_bkg_sm', value=np.nan))

    kin_vars_1, kin_vars_2 = load_kinematics(sample_1), load_kinematics(sample_2)

    sample_1.events = sample_1.events[:int(config['num_events'])]
    kin_vars_1 = kin_vars_1[:int(config['num_events'])]

    sample_2.events = sample_2.events[:int(config['num_events'])]
    kin_vars_2 = kin_vars_2[:int(config['num_events'])]

    true_size_1, true_size_2 = kin_vars_1.shape[0], kin_vars_2.shape[0]

    print(f'Initial base size of {["SIG", "INT", "BKG", "SBI"][component_1.value-1]} set to {int(set_size_1)}. Train and validation data will be {int(true_size_1/2)*2} each after Z mass cuts.')
    print(f'Initial base size of {["SIG", "INT", "BKG", "SBI"][component_2.value-1]} set to {int(set_size_2)}. Train and validation data will be {int(true_size_2/2)*2} each after Z mass cuts.')
    print(f'Total dataset size after filters (per train, val): {int(true_size_1/2) + int(true_size_2/2)}')


    if component_1 != msq.Component.BKG:
        c6_mod = c6.Modifier(baseline = component_1, c6_values = [-5,-1,0,1,5])
        sig_weights, sig_prob = c6_mod.modify(sample=sample_1, c6=config['c6_values'])
    else:
        sig_weights, sig_prob = np.array(sample_1.events.weights)[:,np.newaxis], np.array(sample_1.events.probabilities)[:,np.newaxis]
    
    if component_1 == msq.Component.INT: # TODO: Fix this somehow
        c6_mod = c6.Modifier(baseline = component_1, c6_values = [-5,0,5])
        sig_weights, sig_prob = c6_mod.modify(sample=sample_1, c6=config['c6_values'])
        sig_weights, sig_prob = -1 * sig_weights, -1 * sig_prob

    train_data = datasets.build_dataset_tf(x_arr_sig = kin_vars_1[:int(true_size_1/2)],
                                           x_arr_bkg = kin_vars_2[:int(true_size_2/2)],
                                           param_values = config['c6_values'],
                                           signal_weights = sig_weights[:int(true_size_1/2)],
                                           background_weights = np.array(sample_2.events.weights)[:int(true_size_2/2)],
                                           normalization = 1)

    val_data = datasets.build_dataset_tf(x_arr_sig = kin_vars_1[int(true_size_1/2):],
                                         x_arr_bkg = kin_vars_2[int(true_size_2/2):],
                                         param_values = config['c6_values'],
                                         signal_weights = sig_weights[int(true_size_1/2):],
                                         background_weights = np.array(sample_2.events.weights)[int(true_size_2/2):],
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