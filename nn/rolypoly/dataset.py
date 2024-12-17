from physics.simulation import msq
from physics.hstar import gghzz, c6
from physics.hzz import angles, zpair

import os
import json
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler


def get_component(config):
    component_flag = np.array(config['flags'])[np.where([ (flag in ['sig', 'int', 'sbi', 'bkg']) for flag in config['flags'] ])]
    component = component_flag[0] if component_flag.shape[0] != 0 else 'sbi'
    
    comp_dict = {'sig': msq.Component.SIG,
                 'int': msq.Component.INT,
                 'bkg': msq.Component.BKG,
                 'sbi': msq.Component.SBI}

    return comp_dict[component]

def build_dataset(x_arr, target, weights):
    inputs = tf.cast(x_arr, tf.float32)
    targets = tf.cast(tf.convert_to_tensor(target[:, tf.newaxis]), tf.float32)
    weights = tf.cast(tf.convert_to_tensor(weights[:, tf.newaxis]), tf.float32)

    data = tf.concat([inputs, targets, weights], axis=1)

    return data

def load_sample(config, component):
    if config['num_events'] is None:
        n_i = None
    else:
        n_i = int(config['num_events']*1.2)

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

    return match_comp(config, component, n_i)

def load_kinematics(sample, bounds1=(70,115), bounds2=(70,115), algorithm='leastsquare'):
    z_chooser = zpair.ZPairChooser(bounds1=bounds1, bounds2=bounds2, algorithm=algorithm)

    # calculate_2 uses y4l as additional kinematic
    return angles.calculate_2(*sample.events.filter(z_chooser))

def build(config, seed, strategy=None):
    component = get_component(config)

    sample = load_sample(config, component)

    set_size = sample.events.kinematics.shape[0]

    sample.events.filter(msq.MSQFilter('msq_int_sm', value=0.0))
    sample.events.filter(msq.MSQFilter('msq_int_sm', value=np.nan))

    kin_vars = load_kinematics(sample)

    sample.events = sample.events[:int(config['num_events'])]
    kin_vars = kin_vars[:int(config['num_events'])]

    true_size = kin_vars.shape[0]

    print(f'Initial base size of {["SIG", "INT", "BKG", "SBI"][component.value-1]}(SM) set to {int(set_size)}. Train and validation data will be {int(true_size/2)} each after Z mass cuts.')
    print(f'Total dataset size after filters (per train, val): {int(true_size/2)}')

    c6_mod = c6.Modifier(baseline = component, sample=sample, c6_values = [-5,-1,0,1,5]) if component != msq.Component.INT else c6.Modifier(baseline = component, sample=sample, c6_values = [-1,0,1])
    coeff = c6_mod.coefficients[:, config['coeff']]

    train_data = build_dataset(x_arr = kin_vars[:int(true_size/2)],
                               target = coeff[:int(true_size/2)],
                               weights = sample.events[:int(true_size/2)].probabilities)
    
    val_data = build_dataset(x_arr = kin_vars[int(true_size/2):],
                             target = coeff[int(true_size/2):],
                             weights = sample.events[int(true_size/2):].probabilities)
    
    # The following will scale only kinematics for nonprm and kinematics + c6 for prm
    train_scaler = MinMaxScaler()
    train_data = tf.concat([train_scaler.fit_transform(train_data[:,:-2]), train_data[:,-2:]], axis=1)
    train_data = tf.random.shuffle(train_data, seed=seed)

    val_data = tf.concat([train_scaler.transform(val_data[:,:-2]), val_data[:,-2:]], axis=1)
    val_data = tf.random.shuffle(val_data, seed=seed)

    scaler_config = {'scaler.scale_': train_scaler.scale_.tolist(), 'scaler.min_': train_scaler.min_.tolist()}
    with open('scaler.json', 'w') as scaler_file:
        scaler_file.write(json.dumps(scaler_config, indent=4))

    # Build tf Dataset objects and batch data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data[:,:-2], train_data[:,-2][:,tf.newaxis], train_data[:,-1][:,tf.newaxis]))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data[:,:-2], val_data[:,-2][:,tf.newaxis], val_data[:,-1][:,tf.newaxis]))

    if 'distributed' in config['flags'] and strategy is not None:
        with strategy.scope():
            train_dataset = train_dataset.batch(config['batch_size']*strategy.num_replicas_in_sync)
            val_dataset = val_dataset.batch(config['batch_size']*strategy.num_replicas_in_sync)
    else:
        train_dataset = train_dataset.batch(config['batch_size'])
        val_dataset = val_dataset.batch(config['batch_size'])

    return (train_dataset, val_dataset)