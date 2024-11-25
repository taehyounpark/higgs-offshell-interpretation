import sys
sys.path.append('../')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Nadam 

from hstar import gghzz, c6, msq

from hzz import zpair, angles

from nn import datasets, models

import numpy as np

import os

from sklearn.preprocessing import StandardScaler


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: ', strategy.num_replicas_in_sync)

SEED=373485
GEN=6

# GEN=2, SEED=373485: c6=-20,20 (2001), maxi net (10 layers, 2k nodes each), (added early stopping) epochs=100, train:val = 50:50
# GEN=3, SEED=373485: (shuffled data), lr=0.002 c6=-20,20 (2001), maxi net (10 layers, 2k nodes each), (added early stopping) epochs=100, train:val = 50:50
# GEN=4, SEED=373485: half data (5k), (shuffled data), lr=0.002 c6=-20,20 (2001), maxi net (10 layers, 2k nodes each), (added early stopping) epochs=100, train:val = 50:50
# GEN=5, SEED=373485: data=10k, (shuffled data), lr=0.0001 c6=-20,20 (201), maxi net (10 layers, 2k nodes each), (added early stopping) epochs=150, train:val = 50:50
# GEN=6, SEED=373485: data=1k, (not shuffled data), lr=1e-5 c6=-20,20 (2001), maxi net (10 layers, 2k nodes each), (added early stopping) epochs=150, train:val = 50:50

LEARN_RATE=1e-5
BATCH_SIZE=32
EPOCHS=150
EVENTS_PER_CLASS=99e2

OUTPUT_DIR='../outputs/def'
SAMPLE_DIR='../..'

print(f'Training (classifier) GEN={GEN} with SEED={SEED} on {EVENTS_PER_CLASS} individual events. ML params: lr={LEARN_RATE}, batch_size={BATCH_SIZE}, epochs={EPOCHS}')

sample = gghzz.Process(  
    (1.4783394, SAMPLE_DIR + '/ggZZ2e2m_all_new.csv', 33e2),
    (0.47412769, SAMPLE_DIR + '/ggZZ4e_all_new.csv', 33e2),
    (0.47412769, SAMPLE_DIR + '/ggZZ4m_all_new.csv', 33e2)
)

print('Total events:', sample.events.shape[0])

#base_size = EVENTS_PER_CLASS # for train and validation data each

#fraction = 2*base_size/sample.events.shape[0] # fraction of the dataset that is actually needed

#sample.events = sample.events.sample(frac=fraction, random_state=SEED)

z_chooser = zpair.ZPairChooser(bounds1=(50,115), bounds2=(50,115), algorithm='leastsquare')
l1_1, l2_1, l1_2, l2_2 = sample.events.filter(z_chooser)

kin_variables = angles.calculate(l1_1, l2_1, l1_2, l2_2)

true_size = kin_variables.shape[0]

print(f'Initial base size set to {EVENTS_PER_CLASS}. Train and validation data will be {int(true_size/2)*2} each after Z mass cuts.')

c6_values = np.linspace(-20,20,2001)

c6_mod = c6.Modifier(amplitude_component = msq.Component.SIG, c6_values = [-5,-1,0,1,5])
c6_weights, c6_prob = c6_mod.modify(sample=sample, c6=c6)

train_data = datasets.build_dataset_tf(x_arr = kin_variables[:int(true_size/2)], 
                                       param_values = c6_values, 
                                       signal_weights = c6_weights[:int(true_size/2)], 
                                       background_weights = np.array(sample[msq.Component.SIG].weights)[:int(true_size/2)],
                                       normalization = 1)

val_data = datasets.build_dataset_tf(  x_arr = kin_variables[int(true_size/2):], 
                                       param_values = c6_values, 
                                       signal_weights = c6_weights[int(true_size/2):], 
                                       background_weights = np.array(sample[msq.Component.SIG].weights)[int(true_size/2):],
                                       normalization = 1)


train_scaler = StandardScaler()
train_data = tf.concat([train_scaler.fit_transform(train_data[:,:9]), train_data[:,9:]], axis=1)

train_data = tf.random.shuffle(train_data, seed=SEED)

print('train_data:', train_data, train_data.shape)

val_data = tf.concat([train_scaler.transform(val_data[:,:9]), val_data[:,9:]], axis=1)

val_data = tf.random.shuffle(val_data, seed=SEED)

print('val_data:', val_data, val_data.shape)

print(f'StandardScaler params: \nscaler.mean_ = {train_scaler.mean_.tolist()}\nscaler.var_ = {train_scaler.var_.tolist()}\nscaler.scale_ = {train_scaler.scale_.tolist()}')

model = models.C6_4l_clf_maxi()

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

checkpoint_filepath = OUTPUT_DIR + f'/ckpt/checkpoint.model_{GEN}.tf'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, save_format='tf')
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=15, start_from_epoch=20)

history_callback = model.fit(x=train_data[:,:9], y=train_data[:,9][:,np.newaxis], sample_weight=train_data[:,10][:,np.newaxis], validation_data=(val_data[:,:9], val_data[:,9], val_data[:,10]), batch_size=BATCH_SIZE, callbacks=[model_checkpoint_callback, early_stopping_callback], epochs=EPOCHS, verbose=2)

model.save(OUTPUT_DIR + f'/models/model_{GEN}.tf', save_format='tf')

with open(OUTPUT_DIR + f'/history/history_{GEN}.txt', 'w') as hist_file:
    hist_file.write(str(history_callback.history['loss']))
    hist_file.write(str(history_callback.history['val_loss']))

print(model.summary())
