import numpy as np
import matplotlib.pyplot as plt

import os

from nn.models import ToyClassifier, ToyClassifierPrm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras import Model, backend

print('Number of GPUs available: ', len(tf.config.list_physical_devices('GPU')))

#tf.debugging.set_log_device_placement(True)

SEED=123456
GEN=7

np.random.seed(SEED)

def weights(theta):
    return (1/(1+np.exp(-theta/2))-0.5)

def build_dataset(x_arr, theta_values, normalization=1):
    data = []

    for theta in theta_values:
        # x | theta | y | weight
        data_part_sig = np.append(np.append(x_arr[:,np.newaxis], np.ones(x_arr.shape[0])[:,np.newaxis]*theta, axis=1), np.ones(x_arr.shape[0])[:,np.newaxis], axis=1)
        sig_weights = np.abs(x_arr - (0.5 - weights(theta)))[:,np.newaxis]
        sig_weights *= normalization/np.sum(sig_weights)

        print(f'(theta={theta}) signal weights after: {sig_weights}, sum of weights: {np.sum(sig_weights)}')

        data_part_sig = np.append(data_part_sig, sig_weights, axis=1)

        data_part_bkg = np.append(np.append(x_arr[:,np.newaxis], np.ones(x_arr.shape[0])[:,np.newaxis]*theta, axis=1), np.zeros(x_arr.shape[0])[:,np.newaxis], axis=1)
        bkg_weights = 0.5*np.ones(x_arr.shape[0])[:,np.newaxis]
        bkg_weights *= normalization/np.sum(bkg_weights)

        data_part_bkg = np.append(data_part_bkg, bkg_weights, axis=1)

        data.append(np.append(data_part_sig, data_part_bkg, axis=0))

    data = np.reshape(np.array(data), (np.array(data).shape[0]*np.array(data).shape[1], np.array(data).shape[2]))

    return data


theta_values = np.linspace(-10,10,21)

dataset_size = 40000 # per class and theta value
val_prc = 0.3 # percentage of data used for validation

train_size=int((1-val_prc)*dataset_size)
val_size=int(val_prc*dataset_size)

true_size = dataset_size*2*len(theta_values)

print(f'Total number of events will be: {true_size}, of that {train_size*2*len(theta_values)} training and {val_size*2*len(theta_values)} validation')

x_arr_train = np.append(np.zeros(int(train_size/2)), np.ones(int(train_size/2)))
np.random.shuffle(x_arr_train)

x_arr_val = np.append(np.zeros(int(val_size/2)), np.ones(int(val_size/2)))
np.random.shuffle(x_arr_val)

train_data = build_dataset(x_arr_train, theta_values)
train_data = tf.convert_to_tensor(train_data)
train_data = tf.random.shuffle(train_data, seed=SEED)

print(train_data,train_data.shape)

val_data = build_dataset(x_arr_val, theta_values, normalization=val_prc/(1-val_prc))
val_data = tf.convert_to_tensor(val_data)
val_data = tf.random.shuffle(val_data, seed=SEED)

print(val_data,val_data.shape)

model_prm = ToyClassifierPrm()

optimizer = Nadam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model_prm.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'], weighted_metrics=['binary_accuracy'])


models = []

for theta in theta_values:
    optim = Nadam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    th_mod = ToyClassifier()
    th_mod.compile(optimizer=optim, loss='binary_crossentropy', metrics=['binary_accuracy'], weighted_metrics=['binary_accuracy'])
    models.append(th_mod)


train_datas = []

for theta in theta_values:
    inds = tf.where(train_data[:,1]==theta)
    data_theta = tf.gather(tf.gather(train_data,tf.squeeze(inds)),[0,2,3],axis=1)
    train_datas.append(data_theta)


val_datas = []

for theta in theta_values:
    inds = tf.where(val_data[:,1]==theta)
    data_theta = tf.gather(tf.gather(val_data,tf.squeeze(inds)),[0,2,3],axis=1)
    val_datas.append(data_theta)

history_arr = []

os.makedirs(f'ckpt/{GEN}/', exist_ok=True)

for i in range(len(models)):
    checkpoint_filepath = f'ckpt/{GEN}/checkpoint.model_{i+1}.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True)

    history = models[i].fit(x=train_datas[i][:,0][:,tf.newaxis], y=train_datas[i][:,1][:,tf.newaxis], sample_weight=train_datas[i][:,2][:,tf.newaxis], validation_data=(val_datas[i][:,0][:,tf.newaxis], val_datas[i][:,1][:,tf.newaxis], val_datas[i][:,2][:,tf.newaxis]), epochs=300, verbose=0, callbacks=[model_checkpoint_callback], batch_size=64)
    history_arr.append(history)
    print(f'Model {i+1}: {history.history["loss"][-1]}; {history.history["val_loss"][-1]}')

print(history_arr)

# Save all histories to file
with open(f'mult_history.txt.{GEN}', 'w') as hist_file:
    for hist in history_arr:
        hist_file.write(str(hist.history['loss']))
        hist_file.write(str(hist.history['val_loss']))

# Save all models to file
os.makedirs(f'models/{GEN}/', exist_ok=True)

for i in range(len(models)):
    models[i].save(f'models/{GEN}/model_'+str(i+1)+'_state.keras')

# Save predictions to file
predictions = []
for model in models:
    pred = model.predict(tf.constant([[0.0],[1.0]]), verbose=0)
    predictions.append(pred)

print(predictions)

with open(f'predictions.txt.{GEN}', 'w') as pred_file:
    pred_file.write(str(predictions))

checkpoint_filepath = f'ckpt/{GEN}/checkpoint.model_prm.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True)

prm_hist = model_prm.fit(x=train_data[:,:2], y=train_data[:,2][:,np.newaxis], sample_weight=train_data[:,3][:,np.newaxis], validation_data=(val_data[:,:2][:,tf.newaxis], val_data[:,2][:,tf.newaxis], val_data[:,3][:,tf.newaxis]), epochs=300, verbose=0, callbacks=[model_checkpoint_callback], batch_size=64)

model_prm.save(f'models/{GEN}/model_prm_state.keras')

print(prm_hist)

# Save prm hist to file
with open(f'prm_history.txt.{GEN}', 'w') as hist_file:
    hist_file.write(str(prm_hist.history['loss']))
    hist_file.write(str(prm_hist.history['val_loss']))

# Save prm predictions to file
prm_predictions = []
for theta in theta_values:
    pred = model_prm.predict(tf.constant([[0.0, theta],[1.0, theta]]), verbose=0)
    prm_predictions.append(pred)

print(prm_predictions)

with open(f'prm_prediction.txt.{GEN}', 'w') as pred_file:
    pred_file.write(str(prm_predictions))