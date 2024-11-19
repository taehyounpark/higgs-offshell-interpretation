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
GEN=9

# GEN=2, SEED=373485: data: 1M, c6=20, maxi net (10 layers, 2k nodes each), (added early stopping) epochs=100, train:val = 50:50
# GEN=3, SEED=373485: data: 1M, lr=0.003, c6=20, maxi net (10 layers, 2k nodes each), (added early stopping) epochs=150, train:val = 50:50
# GEN=4, SEED=373485: data: 2M, lr=0.003, c6=20, maxi net (10 layers, 2k nodes each), (added early stopping) epochs=150, train:val = 50:50
# GEN=5, (was saved under 4) SEED=373485: (shuffled train, val) data: 2M, lr=0.002, c6=20, maxi net (10 layers, 2k nodes each), (no early stopping) epochs=250, train:val = 50:50
# GEN=6, SEED=373485: data: 200k, lr=0.005, c6=-10, maxi net (10 layers, 2k nodes each), (early stopping after 20 epochs) epochs=250, train:val = 50:50
# GEN=7, SEED=373485: data: 200k, lr=0.0001, c6=-10, maxi net (10 layers, 2k nodes each), (early stopping after 20 epochs) epochs=50, train:val = 50:50, batch_size=16
# GEN=8,(continuing from GEN 7 checkpoint) SEED=373485: data: 1M, lr=0.0001, c6=-10, maxi net (10 layers, 2k nodes each), (early stopping after 20 epochs) epochs=100, train:val = 50:50, batch_size=32
# GEN=9,(same as GEN 7, but higher complexity, lower batch size) SEED=373485: data: 200k, lr=0.0001, c6=-10, big net (10 layers, 2.5k nodes each), (early stopping after 20 epochs) epochs=50, train:val = 50:50, batch_size=8


LEARN_RATE=1e-4
BATCH_SIZE=16
EPOCHS=50
EVENTS_PER_CLASS=100000

print(f'Training (classifier-single) GEN={GEN} with SEED={SEED} on {EVENTS_PER_CLASS} individual events. ML params: lr={LEARN_RATE}, batch_size={BATCH_SIZE}, epochs={EPOCHS}')

OUTPUT_DIR='../outputs/single'
SAMPLE_DIR='../..'

sample = gghzz.Process(  
    (1.4783394, SAMPLE_DIR + '/ggZZ2e2m_all_new.csv', 1e6),
    (0.47412769, SAMPLE_DIR + '/ggZZ4e_all_new.csv', 1e6),
    (0.47412769, SAMPLE_DIR + '/ggZZ4m_all_new.csv', 1e6)
)

print('Total events:', sample.events.kinematics.shape[0])

base_size = EVENTS_PER_CLASS # for train and validation data each

fraction = 2*base_size/sample.events.shape[0] # fraction of the dataset that is actually needed

sample.events = sample.events.sample(frac=fraction, random_state=SEED) # shuffle dataset and take out the specified fraction

z_chooser = zpair.ZPairChooser(bounds1=(50,115), bounds2=(50,115), algorithm='leastsquare')
l1_1, l2_1, l1_2, l2_2 = sample.events.filter(z_chooser)

kin_variables  = angles.calculate(l1_1, l2_1, l1_2, l2_2)

true_size = kin_variables.shape[0]

print(f'Initial base size set to {base_size}. Train and validation data will be {int(true_size/2)*2} each after Z mass cuts.')

c6 = [-10]

c6_mod = c6.Modifier(amplitude_component = msq.Component.SBI, c6_values = [-5,-1,0,1,5])
c6_weights, c6_prob = c6_mod.modify(sample=sample, c6=c6)

train_data = datasets.build_dataset_tf(x_arr = kin_variables[:int(true_size/2)], 
                                       param_values = c6, 
                                       signal_weights = c6_weights[:int(true_size/2)], 
                                       background_weights = np.array(sample.events['wt'])[:int(true_size/2)],
                                       normalization = 1)

val_data = datasets.build_dataset_tf(  x_arr = kin_variables[int(true_size/2):], 
                                       param_values = c6, 
                                       signal_weights = c6_weights[int(true_size/2):], 
                                       background_weights = np.array(sample.events['wt'])[int(true_size/2):],
                                       normalization = 1)


train_scaler = StandardScaler()
train_data = tf.concat([train_scaler.fit_transform(train_data[:,:8]), train_data[:,8:]], axis=1)

train_data = tf.random.shuffle(train_data, seed=SEED)

print('train_data:', train_data, train_data.shape)

val_data = tf.concat([train_scaler.transform(val_data[:,:8]), val_data[:,8:]], axis=1)

val_data = tf.random.shuffle(val_data, seed=SEED)

print('val_data:', val_data, val_data.shape)

print(f'StandardScaler params: mu={train_scaler.mean_.tolist()}, variance={train_scaler.var_.tolist()}, scale={train_scaler.scale_.tolist()}')

model = models.C6_4l_clf_big_nonprm()

optimizer = Nadam(
    learning_rate=LEARN_RATE,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'], weighted_metrics=['binary_accuracy'])

# Load old checkpoint
#model = keras.models.load_model(OUTPUT_DIR + f'/ckpt/checkpoint.model_7.tf', custom_objects={'C6_4l_clf_maxi_nonprm': C6_4l_clf_maxi_nonprm, 'swish_activation': swish_activation})

os.makedirs(OUTPUT_DIR + '/ckpt/', exist_ok=True)
os.makedirs(OUTPUT_DIR + '/models/', exist_ok=True)
os.makedirs(OUTPUT_DIR + '/history/', exist_ok=True)

checkpoint_filepath = OUTPUT_DIR + f'/ckpt/checkpoint.model_{GEN}.tf'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, save_format='tf')
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, start_from_epoch=20)

history_callback = model.fit(x=train_data[:,:8], y=train_data[:,8][:,np.newaxis], sample_weight=train_data[:,9][:,np.newaxis], validation_data=(val_data[:,:8], val_data[:,8], val_data[:,9]), batch_size=BATCH_SIZE, callbacks=[model_checkpoint_callback], epochs=EPOCHS, verbose=2)

model.save(OUTPUT_DIR + f'/models/model_{GEN}.tf', save_format='tf')

with open(OUTPUT_DIR + f'/history/history_{GEN}.txt', 'w') as hist_file:
    hist_file.write(str(history_callback.history['loss']))
    hist_file.write(str(history_callback.history['val_loss']))

print(model.summary())
