import numpy as np
import tensorflow as tf 


def build_dataset_tf(x_arr_sig, x_arr_bkg, param_values, signal_weights, background_weights, normalization=1):
    data = []

    non_prm=False

    if np.isscalar(param_values):
        param_values = [param_values]
        non_prm=True
    elif len(param_values) == 1:
        non_prm=True

    for i in range(len(param_values)):
        param = param_values[i]
        # x | param | y | weight
        if not non_prm:
            if len(x_arr_sig.shape) == 1:
                data_part_sig = tf.concat([x_arr_sig[:,tf.newaxis], tf.ones(x_arr_sig.shape[0])[:,tf.newaxis]*param, tf.ones(x_arr_sig.shape[0])[:,tf.newaxis]], axis=1)
            else:
                data_part_sig = tf.concat([x_arr_sig, tf.ones(x_arr_sig.shape[0])[:,tf.newaxis]*param, tf.ones(x_arr_sig.shape[0])[:,tf.newaxis]], axis=1)

            sig_weights = tf.cast(signal_weights.T[i][:,tf.newaxis], tf.float32)
            if normalization is not None:
                sig_weights *= normalization/tf.reduce_sum(sig_weights)

            data_part_sig = tf.concat([data_part_sig, sig_weights], axis=1)


            if len(x_arr_bkg.shape) == 1:
                data_part_bkg = tf.concat([x_arr_bkg[:,tf.newaxis], tf.ones(x_arr_bkg.shape[0])[:,tf.newaxis]*param, tf.zeros(x_arr_bkg.shape[0])[:,tf.newaxis]], axis=1)
            else:
                data_part_bkg = tf.concat([x_arr_bkg, tf.ones(x_arr_bkg.shape[0])[:,tf.newaxis]*param, tf.zeros(x_arr_bkg.shape[0])[:,tf.newaxis]], axis=1)
            
            bkg_weights = tf.cast(background_weights[:,tf.newaxis], tf.float32)
            if normalization is not None:
                bkg_weights *= normalization/tf.reduce_sum(bkg_weights)

            data_part_bkg = tf.concat([data_part_bkg, bkg_weights], axis=1)

            data.append(tf.concat([data_part_sig, data_part_bkg], axis=0))
        else:
            if len(x_arr_sig.shape) == 1:
                data_part_sig = tf.concat([x_arr_sig[:,tf.newaxis], tf.ones(x_arr_sig.shape[0])[:,tf.newaxis]], axis=1)
            else:
                data_part_sig = tf.concat([x_arr_sig, tf.ones(x_arr_sig.shape[0])[:,tf.newaxis]], axis=1)

            sig_weights = tf.cast(signal_weights.T[i][:,tf.newaxis], tf.float32)
            if normalization is not None:
                sig_weights *= normalization/tf.reduce_sum(sig_weights)

            data_part_sig = tf.concat([data_part_sig, sig_weights], axis=1)


            if len(x_arr_bkg.shape) == 1:
                data_part_bkg = tf.concat([x_arr_bkg[:,tf.newaxis], tf.zeros(x_arr_bkg.shape[0])[:,tf.newaxis]], axis=1)
            else:
                data_part_bkg = tf.concat([x_arr_bkg, tf.zeros(x_arr_bkg.shape[0])[:,tf.newaxis]], axis=1)
            
            bkg_weights = tf.cast(background_weights[:,tf.newaxis], tf.float32)
            if normalization is not None:
                bkg_weights *= normalization/tf.reduce_sum(bkg_weights)

            data_part_bkg = tf.concat([data_part_bkg, bkg_weights], axis=1)

            data.append(tf.concat([data_part_sig, data_part_bkg], axis=0))

    data = tf.reshape(tf.convert_to_tensor(data), (tf.convert_to_tensor(data).shape[0]*tf.convert_to_tensor(data).shape[1], tf.convert_to_tensor(data).shape[2]))

    return data

def build_dataset_random(x_arr_sig, x_arr_bkg, param_values, signal_probabilities, background_probabilities, seed=None):
    non_prm=False

    if np.isscalar(param_values):
        param_values = [param_values]
        non_prm=True
    elif len(param_values) == 1:
        non_prm=True
    
    rng = np.random.default_rng(seed=seed)
    param_per_event = rng.choice(param_values, x_arr_sig.shape[0])

    indices_per_event = np.zeros_like(param_per_event, dtype=np.int32)

    for i in range(len(param_values)):
        indices_per_event[np.where(param_per_event == param_values[i])] = int(i)

    sig_probabilities = signal_probabilities[np.arange(param_per_event.shape[0]),indices_per_event]

    # signal weights should be renormalized per param value
    for i in range(len(param_values)):
        sig_probabilities[np.where(param_per_event == param_values[i])] /= np.sum(sig_probabilities[np.where(param_per_event == param_values[i])])
    
    sig_probabilities /= np.array(param_values).shape[0]

    sig_data = tf.concat([tf.convert_to_tensor(x_arr_sig, dtype=tf.float32), tf.convert_to_tensor(param_per_event, dtype=tf.float32)[:,tf.newaxis], tf.ones(x_arr_sig.shape[0])[:,tf.newaxis], tf.convert_to_tensor(sig_probabilities, dtype=tf.float32)[:,tf.newaxis]], axis=1)

    bkg_data = tf.concat([tf.convert_to_tensor(x_arr_bkg, dtype=tf.float32), tf.convert_to_tensor(param_per_event, dtype=tf.float32)[:,tf.newaxis], tf.zeros(x_arr_sig.shape[0])[:,tf.newaxis], tf.convert_to_tensor(background_probabilities, dtype=tf.float32)[:,tf.newaxis]], axis=1)

    return tf.concat([sig_data, bkg_data], axis=0)
