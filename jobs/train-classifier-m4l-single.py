import sys
sys.path.append('../')

from hstar import gghzz, c6, msq

from hzz import zpair, angles

from nn import datasets, models

import numpy as np

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Nadam

from sklearn.preprocessing import StandardScaler

print('Number of GPUs available: ', len(tf.config.list_physical_devices('GPU')))

SEED=373485
GEN=2

LEARN_RATE=1e-4
BATCH_SIZE=64
EPOCHS=100
EVENTS_PER_CLASS=3000

print(f'Training (classifier-m4l-single) GEN={GEN} with SEED={SEED} on {EVENTS_PER_CLASS} individual events. ML params: lr={LEARN_RATE}, batch_size={BATCH_SIZE}, epochs={EPOCHS}')

OUTPUT_DIR='../outputs/m4l/single'
SAMPLE_DIR='../..'

sample = gghzz.Process(  
    (1.4783394, SAMPLE_DIR + '/ggZZ2e2m_all_new.csv', 1e6),
    (0.47412769, SAMPLE_DIR + '/ggZZ4e_all_new.csv', 1e6),
    (0.47412769, SAMPLE_DIR + '/ggZZ4m_all_new.csv', 1e6)
)

base_size = EVENTS_PER_CLASS # for train and validation data each

fraction = 2*base_size/sample.events.shape[0] # fraction of the dataset that is actually needed

sample.events = sample.events.sample(frac=fraction, random_state=SEED)

z_chooser = zpair.ZPairChooser(bounds1=(50,115), bounds2=(50,115), algorithm='leastsquare')
l1_1, l2_1, l1_2, l2_2 = sample.events.filter(z_chooser)

kin_variables = angles.calculate(l1_1, l2_1, l1_2, l2_2)

true_size = kin_variables.shape[0]

c6 = [-10]

c6_mod = c6.Modifier(amplitude_component = msq.Component.SBI, c6_values = [-5,-1,0,1,5])
c6_weights, c6_prob = c6_mod.modify(sample=sample, c6=c6)

train_data = datasets.build_dataset_tf(x_arr = kin_variables[:int(true_size/2), -1], 
                                       param_values = c6, 
                                       signal_weights = c6_weights[:int(true_size/2)], 
                                       background_weights = np.array(sample.events.weights)[:int(true_size/2)],
                                       normalization = 1)

val_data = datasets.build_dataset_tf(  x_arr = kin_variables[int(true_size/2):, -1], 
                                       param_values = c6, 
                                       signal_weights = c6_weights[int(true_size/2):], 
                                       background_weights = np.array(sample.events.weights)[int(true_size/2):],
                                       normalization = 1)

train_scaler = StandardScaler()
train_data = tf.concat([train_scaler.fit_transform(train_data[:,0][:,tf.newaxis]), train_data[:,1:]], axis=1)

train_data = tf.random.shuffle(train_data, seed=SEED)

print(train_data, train_data.shape)

val_data = tf.concat([train_scaler.transform(val_data[:,0][:,tf.newaxis]), val_data[:,1:]], axis=1)

val_data = tf.random.shuffle(val_data, seed=SEED)

print(val_data, val_data.shape)

print(f'StandardScaler params: mu={train_scaler.mean_.tolist()}, variance={train_scaler.var_.tolist()}, scale={train_scaler.scale_.tolist()}')

model = models.C6_4l_clf_mini()

optimizer = Nadam(
    learning_rate=LEARN_RATE,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'], weighted_metrics=['binary_accuracy'])

os.makedirs(OUTPUT_DIR + '/ckpt/', exist_ok=True)
os.makedirs(OUTPUT_DIR + '/models/', exist_ok=True)
os.makedirs(OUTPUT_DIR + '/history/', exist_ok=True)

checkpoint_filepath = OUTPUT_DIR + f'/ckpt/checkpoint.model_{GEN}_m4l.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True)

history_callback = model.fit(x=train_data[:,0][:,np.newaxis], y=train_data[:,1][:,np.newaxis], sample_weight=train_data[:,2][:,np.newaxis], validation_split=0.3, batch_size=BATCH_SIZE, callbacks=[model_checkpoint_callback], epochs=EPOCHS, verbose=2)

model.save(OUTPUT_DIR + f'/models/model_{GEN}_m4l.keras')

with open(OUTPUT_DIR + f'/history/history_{GEN}_m4l.txt', 'w') as hist_file:
    hist_file.write(str(history_callback.history['loss']))
    hist_file.write(str(history_callback.history['val_loss']))
