from hstar import process
from hstar import trilinear

from hzz import zcandidate
from hzz import angles

from nn import datasets
from nn import models

import numpy as np

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Nadam # type: ignore

print('Number of GPUs available: ', len(tf.config.list_physical_devices('GPU')))

SEED=373485
GEN=1


sample = process.Sample(weight='wt', 
    amplitude = process.Basis.SBI, components = {
    process.Basis.SBI: 'msq_sbi_sm',
    process.Basis.SIG: 'msq_sig_sm',
    process.Basis.BKG: 'msq_bkg_sm',
    process.Basis.INT: 'msq_int_sm'
  })

sample.open(csv = [
  '../ggZZ4e_all_new.csv',
  '../ggZZ4m_all_new.csv',
  '../ggZZ2e2m_all_new.csv'
  ], xs=[1.4783394, 0.47412769, 0.47412769], lumi=3000., k=1.83, nrows=1000000
)

sample.events = sample.events.sample(frac=1).reset_index(drop=True)

zmasses = zcandidate.ZmassPairChooser(sample)
leptons = zmasses.find_Z()

kin_variables, filter_indices = angles.calculate(leptons.T[0], leptons.T[1], leptons.T[2], leptons.T[3])

kin_variables = kin_variables[filter_indices]
sample.events = sample.events.take(indices=filter_indices)

c6 = 10

c6_mod = trilinear.Modifier(c6_values = [-5,-1,0,1,5], c6_amplitudes = ['msq_sbi_c6_6', 'msq_sbi_c6_10', 'msq_sbi_c6_11', 'msq_sbi_c6_12', 'msq_sbi_c6_16'])
c6_weights = c6_mod.modify(sample=sample, c6=c6)

train_data = datasets.build_dataset_tf(x_arr = kin_variables, 
                                       param_values = [c6], 
                                       signal_weights = c6_weights, 
                                       background_weights = np.array(sample.events['wt']),
                                       normalization = 1)

train_data = tf.random.shuffle(train_data, SEED)

print(train_data, train_data.shape)

model = models.C6_4l_clf_reduced_nonprm()

optimizer = Nadam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'], weighted_metrics=['binary_accuracy'])

os.makedirs('outputs_single/ckpt/', exist_ok=True)
os.makedirs('outputs_single/models/', exist_ok=True)
os.makedirs('outputs_single/history/', exist_ok=True)

checkpoint_filepath = f'outputs_single/ckpt/checkpoint.model_{GEN}.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True)

history_callback = model.fit(x=train_data[:,:8], y=train_data[:,8][:,np.newaxis], sample_weight=train_data[:,9][:,np.newaxis], validation_split=0.3, batch_size=64, callbacks=[model_checkpoint_callback], epochs=100, verbose=2)


model.save(f'outputs_single/models/model_{GEN}.keras')

with open(f'outputs_single/history/history_{GEN}.txt', 'w') as hist_file:
    hist_file.write(str(history_callback.history['loss']))
    hist_file.write(str(history_callback.history['val_loss']))
