import sys
sys.path.append('../')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Nadam 

from hstar import process
from hstar import trilinear

from hzz import zcandidate
from hzz import angles

from nn import datasets
from nn import models

import numpy as np

import os

from sklearn.preprocessing import StandardScaler


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: ', strategy.num_replicas_in_sync)

SEED=373485
GEN=4

# GEN=2, SEED=373485: data: 1M, c6=20, maxi net (10 layers, 2k nodes each), (added early stopping) epochs=100, train:val = 50:50
# GEN=3, SEED=373485: data: 1M, lr=0.03, c6=20, maxi net (10 layers, 2k nodes each), (added early stopping) epochs=150, train:val = 50:50
# GEN=4, SEED=373485: data: 2M, lr=0.03, c6=20, maxi net (10 layers, 2k nodes each), (added early stopping) epochs=150, train:val = 50:50

OUTPUT_DIR='../outputs/single'
SAMPLE_DIR='../..'

sample = process.Sample(weight='wt', 
    amplitude = process.Basis.SBI, components = {
    process.Basis.SBI: 'msq_sbi_sm',
    process.Basis.SIG: 'msq_sig_sm',
    process.Basis.BKG: 'msq_bkg_sm',
    process.Basis.INT: 'msq_int_sm'
  })

sample.open(csv = [
  SAMPLE_DIR + '/ggZZ2e2m_all_new.csv',
  SAMPLE_DIR + '/ggZZ4e_all_new.csv',
  SAMPLE_DIR + '/ggZZ4m_all_new.csv'
  ], xs=[1.4783394, 0.47412769, 0.47412769], lumi=3000., k=1.83
)

print('Total events:', sample.events.shape[0])

base_size = 1000000 # for train and validation data each

fraction = 2*base_size/sample.events.shape[0] # fraction of the dataset that is actually needed

sample.events = sample.events.sample(frac=fraction, random_state=SEED, ignore_index=True) # shuffle dataset and take out the specified fraction

zmasses = zcandidate.ZmassPairChooser(sample)
leptons = zmasses.find_Z()

kin_variables  = angles.calculate(leptons.T[0], leptons.T[1], leptons.T[2], leptons.T[3])

true_size = kin_variables.shape[0]

print(f'Initial base size set to {base_size}. Train and validation data will be {int(true_size/2)*2} each after Z mass cuts.')

c6 = 20

c6_mod = trilinear.Modifier(c6_values = [-5,-1,0,1,5], c6_amplitudes = ['msq_sbi_c6_6', 'msq_sbi_c6_10', 'msq_sbi_c6_11', 'msq_sbi_c6_12', 'msq_sbi_c6_16'])
c6_weights = c6_mod.modify(sample=sample, c6=c6)

train_data = datasets.build_dataset_tf(x_arr = kin_variables[:int(true_size/2)], 
                                       param_values = [c6], 
                                       signal_weights = c6_weights[:int(true_size/2)], 
                                       background_weights = np.array(sample.events['wt'])[:int(true_size/2)],
                                       normalization = 1)

val_data = datasets.build_dataset_tf(  x_arr = kin_variables[int(true_size/2):], 
                                       param_values = [c6], 
                                       signal_weights = c6_weights[int(true_size/2):], 
                                       background_weights = np.array(sample.events['wt'])[int(true_size/2):],
                                       normalization = 1)



train_scaler = StandardScaler()
train_data = tf.concat([train_scaler.fit_transform(train_data[:,:8]), train_data[:,8:]], axis=1)

print('train_data:', train_data, train_data.shape)

val_scaler = StandardScaler()
val_data = tf.concat([val_scaler.fit_transform(val_data[:,:8]), val_data[:,8:]], axis=1)

print('val_data:', val_data, val_data.shape)

model = models.C6_4l_clf_maxi_nonprm()

optimizer = Nadam(
    learning_rate=0.003,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'], weighted_metrics=['binary_accuracy'])

os.makedirs(OUTPUT_DIR + '/ckpt/', exist_ok=True)
os.makedirs(OUTPUT_DIR + '/models/', exist_ok=True)
os.makedirs(OUTPUT_DIR + '/history/', exist_ok=True)

checkpoint_filepath = OUTPUT_DIR + f'/ckpt/checkpoint.model_{GEN}.tf'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, save_format='tf')
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=8, start_from_epoch=20)

history_callback = model.fit(x=train_data[:,:8], y=train_data[:,8][:,np.newaxis], sample_weight=train_data[:,9][:,np.newaxis], validation_data=(val_data[:,:8], val_data[:,8], val_data[:,9]), batch_size=64, callbacks=[model_checkpoint_callback, early_stopping_callback], epochs=150, verbose=2)

model.save(OUTPUT_DIR + f'/models/model_{GEN}.tf', save_format='tf')

with open(OUTPUT_DIR + f'/history/history_{GEN}.txt', 'w') as hist_file:
    hist_file.write(str(history_callback.history['loss']))
    hist_file.write(str(history_callback.history['val_loss']))

print(model.summary())
