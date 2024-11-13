import numpy as np
import tensorflow as tf 

def build_dataset(x_arr, param_values, signal_weights, background_weights, normalization=1):
    data = []

    non_prm=False

    if np.isscalar(param_values):
        param_values = [param_values]
        non_prm=True
        print('Non param')
    elif len(param_values) == 1:
        non_prm=True
        print('Non param')

    for i in range(len(param_values)):
        param = param_values[i]
        # x | param | y | weight
        if len(x_arr.shape) == 1:
            data_part_sig = np.append(np.append(x_arr[:,np.newaxis], np.ones(x_arr.shape[0])[:,np.newaxis]*param, axis=1), np.ones(x_arr.shape[0])[:,np.newaxis], axis=1)
        else:
            data_part_sig = np.append(np.append(x_arr, np.ones(x_arr.shape[0])[:,np.newaxis]*param, axis=1), np.ones(x_arr.shape[0])[:,np.newaxis], axis=1)

        sig_weights = signal_weights.T[i][:,np.newaxis]
        sig_weights *= normalization/np.sum(sig_weights)

        data_part_sig = np.append(data_part_sig, sig_weights, axis=1)


        if len(x_arr.shape) == 1:
            data_part_bkg = np.append(np.append(x_arr[:,np.newaxis], np.ones(x_arr.shape[0])[:,np.newaxis]*param, axis=1), np.zeros(x_arr.shape[0])[:,np.newaxis], axis=1)
        else:
            data_part_bkg = np.append(np.append(x_arr, np.ones(x_arr.shape[0])[:,np.newaxis]*param, axis=1), np.zeros(x_arr.shape[0])[:,np.newaxis], axis=1)        
        
        bkg_weights = background_weights[:,np.newaxis]
        bkg_weights *= normalization/np.sum(bkg_weights)

        data_part_bkg = np.append(data_part_bkg, bkg_weights, axis=1)

        data.append(np.append(data_part_sig, data_part_bkg, axis=0))

    data = np.reshape(np.array(data), (np.array(data).shape[0]*np.array(data).shape[1], np.array(data).shape[2]))

    return data


def build_dataset_tf(x_arr, param_values, signal_weights, background_weights, normalization=1):
    data = []

    non_prm=False

    if np.isscalar(param_values):
        param_values = [param_values]
        non_prm=True
        print('Non param')
    elif len(param_values) == 1:
        non_prm=True
        print('Non param')

    for i in range(len(param_values)):
        param = param_values[i]
        # x | param | y | weight
        if not non_prm:
            if len(x_arr.shape) == 1:
                data_part_sig = tf.concat([tf.concat([x_arr[:,tf.newaxis], tf.ones(x_arr.shape[0])[:,tf.newaxis]*param], axis=1), tf.ones(x_arr.shape[0])[:,tf.newaxis]], axis=1)
            else:
                data_part_sig = tf.concat([tf.concat([x_arr, tf.ones(x_arr.shape[0])[:,tf.newaxis]*param], axis=1), tf.ones(x_arr.shape[0])[:,tf.newaxis]], axis=1)

            sig_weights = tf.cast(signal_weights.T[i][:,tf.newaxis], tf.float32)
            sig_weights *= normalization/tf.reduce_sum(sig_weights)

            data_part_sig = tf.concat([data_part_sig, sig_weights], axis=1)


            if len(x_arr.shape) == 1:
                data_part_bkg = tf.concat([tf.concat([x_arr[:,tf.newaxis], tf.ones(x_arr.shape[0])[:,tf.newaxis]*param], axis=1), tf.zeros(x_arr.shape[0])[:,tf.newaxis]], axis=1)
            else:
                data_part_bkg = tf.concat([tf.concat([x_arr, tf.ones(x_arr.shape[0])[:,tf.newaxis]*param], axis=1), tf.zeros(x_arr.shape[0])[:,tf.newaxis]], axis=1)
            
            bkg_weights = tf.cast(background_weights[:,tf.newaxis], tf.float32)
            bkg_weights *= normalization/tf.reduce_sum(bkg_weights)

            data_part_bkg = tf.concat([data_part_bkg, bkg_weights], axis=1)

            data.append(tf.concat([data_part_sig, data_part_bkg], axis=0))
        else:
            if len(x_arr.shape) == 1:
                data_part_sig = tf.concat([x_arr[:,tf.newaxis], tf.ones(x_arr.shape[0])[:,tf.newaxis]], axis=1)
            else:
                data_part_sig = tf.concat([x_arr, tf.ones(x_arr.shape[0])[:,tf.newaxis]], axis=1)

            sig_weights = tf.cast(signal_weights.T[i][:,tf.newaxis], tf.float32)
            sig_weights *= normalization/tf.reduce_sum(sig_weights)

            data_part_sig = tf.concat([data_part_sig, sig_weights], axis=1)


            if len(x_arr.shape) == 1:
                data_part_bkg = tf.concat([x_arr[:,tf.newaxis], tf.zeros(x_arr.shape[0])[:,tf.newaxis]], axis=1)
            else:
                data_part_bkg = tf.concat([x_arr, tf.zeros(x_arr.shape[0])[:,tf.newaxis]], axis=1)
            
            bkg_weights = tf.cast(background_weights[:,tf.newaxis], tf.float32)
            bkg_weights *= normalization/tf.reduce_sum(bkg_weights)

            data_part_bkg = tf.concat([data_part_bkg, bkg_weights], axis=1)

            data.append(tf.concat([data_part_sig, data_part_bkg], axis=0))

    data = tf.reshape(tf.convert_to_tensor(data), (tf.convert_to_tensor(data).shape[0]*tf.convert_to_tensor(data).shape[1], tf.convert_to_tensor(data).shape[2]))

    return data