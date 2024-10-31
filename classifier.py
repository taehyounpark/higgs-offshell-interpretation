from hstar import process
from hstar import trilinear
from hstar import datacuts

from hzz import zcandidate
from hzz import angles

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import Model, backend


sample = process.Sample(weight='wt', 
    amplitude = process.Basis.SBI, components = {
    process.Basis.SBI: 'msq_sbi_sm',
    process.Basis.SIG: 'msq_sig_sm',
    process.Basis.BKG: 'msq_bkg_sm',
    process.Basis.INT: 'msq_int_sm'
  })

sample.open(csv = [
  '/home/max/Uni/WS24/BachelorArbeit/ggZZ4e_all.csv',
  '/home/max/Uni/WS24/BachelorArbeit/ggZZ4m_all.csv',
  '/home/max/Uni/WS24/BachelorArbeit/ggZZ2e2m_all.csv'
  ], xs=[1.4783394, 0.47412769, 0.47412769], lumi=3000., k=1.83, nrows=100000
)

zmasses = zcandidate.ZmassPairChooser(sample)
leptons = zmasses.find_Z()

kin_variables = angles.calculate(leptons.T[0], leptons.T[1], leptons.T[2], leptons.T[3])

c6_values = np.linspace(-10,10,21)

c6_mod = trilinear.Modifier(c6_values = [-5,-1,0,1,5], c6_amplitudes = ['msq_sbi_c6_6', 'msq_sbi_c6_10', 'msq_sbi_c6_11', 'msq_sbi_c6_12', 'msq_sbi_c6_16'])
c6_weights = c6_mod.modify(sample=sample, c6=c6_values)

data = []

for i in range(len(c6_values)):
    c6 = c6_values[i]

    # {kin}, c6, label, weight
    data_part_sig = np.append(np.append(kin_variables, np.ones((kin_variables.shape[0],1))*c6, axis=1), np.ones((kin_variables.shape[0],1)), axis=1)
    sig_weights = c6_weights.T[i][:,np.newaxis]
    sig_weights *= 1/np.sum(sig_weights)

    data_part_sig = np.append(data_part_sig, sig_weights, axis=1)

    data_part_bkg = np.append(np.append(kin_variables, np.ones((kin_variables.shape[0],1))*c6, axis=1), np.zeros((kin_variables.shape[0],1)), axis=1)
    bkg_weights = np.array(sample.events['wt'])[:,np.newaxis]
    bkg_weights *= 1/np.sum(bkg_weights)

    data_part_bkg = np.append(data_part_bkg, bkg_weights, axis=1)

    data_part = np.append(data_part_sig, data_part_bkg, axis=0)

    data.append(data_part)


data = np.reshape(np.array(data), (np.array(data).shape[0]*np.array(data).shape[1], np.array(data).shape[2]))


def swish_activation(x, b=1):
    return x*backend.sigmoid(b*x)

get_custom_objects().update({'swish_activation': Activation(swish_activation)})


class C6_4l_clf(Model):
    def __init__(self):
        super().__init__()

        swish = Activation(swish_activation, name='Swish')

        self.dense1 = Dense(1000, activation=swish, input_dim=8, kernel_initializer='he_normal')
        self.dense2 = Dense(1000, activation=swish, kernel_initializer='he_normal')
        self.dense3 = Dense(1000, activation=swish, kernel_initializer='he_normal')
        self.dense4 = Dense(1000, activation=swish, kernel_initializer='he_normal')
        self.dense5 = Dense(1000, activation=swish, kernel_initializer='he_normal')
        self.out = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.out(x)
    
model = C6_4l_clf()

optimizer = Nadam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'], weighted_metrics=['binary_accuracy'])