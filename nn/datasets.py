import numpy as np

def build_dataset(x_arr, param_values, signal_weights, background_weights, normalization=1):
    data = []

    for i in range(len(param_values)):
        param = param_values[i]
        # x | param | y | weight
        if len(x_arr.shape) == 1:
            data_part_sig = np.append(np.append(x_arr[:,np.newaxis], np.ones(x_arr.shape[0])[:,np.newaxis]*param, axis=1), np.ones(x_arr.shape[0])[:,np.newaxis], axis=1)
        else:
            data_part_sig = np.append(np.append(x_arr, np.ones(x_arr.shape[0])[:,np.newaxis]*param, axis=1), np.ones(x_arr.shape[0])[:,np.newaxis], axis=1)

        print(data_part_sig, data_part_sig.shape)

        sig_weights = signal_weights.T[i][:,np.newaxis]
        sig_weights *= normalization/np.sum(sig_weights)

        print(sig_weights, sig_weights.shape)

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