import sys
sys.path.append('../')

import tensorflow as tf
from tensorflow import keras

from nn.models import C6_4l_clf_maxi_nonprm, swish_activation
from nn import datasets
from hstar import process, trilinear
from hzz import zcandidate, angles

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

SEED=373485
GEN=7
OUTPUT_DIR='../outputs/single'
SAMPLE_DIR='../..'

'''sample = process.Sample(weight='wt', 
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

base_size = 100000 # for train and validation data each

fraction = 2*base_size/sample.events.shape[0] # fraction of the dataset that is actually needed

sample.events = sample.events.sample(frac=fraction, random_state=SEED, ignore_index=True) # shuffle dataset and take out the specified fraction

zmasses = zcandidate.ZmassPairChooser(sample)
leptons = zmasses.find_Z()

kin_variables  = angles.calculate(leptons.T[0], leptons.T[1], leptons.T[2], leptons.T[3])

true_size = kin_variables.shape[0]

print(f'Initial base size set to {base_size}. Train and validation data will be {int(true_size/2)*2} each after Z mass cuts.')

c6 = [-10]

c6_mod = trilinear.Modifier(c6_values = [-5,-1,0,1,5], c6_amplitudes = ['msq_sbi_c6_6', 'msq_sbi_c6_10', 'msq_sbi_c6_11', 'msq_sbi_c6_12', 'msq_sbi_c6_16'])
c6_weights = c6_mod.modify(sample=sample, c6=c6)

train_data = datasets.build_dataset_tf(x_arr = kin_variables[:int(true_size/2)], 
                                       param_values = c6, 
                                       signal_weights = c6_weights[:int(true_size/2)], 
                                       background_weights = np.array(sample.events['wt'])[:int(true_size/2)],
                                       normalization = 1)
train_scaler = StandardScaler()
train_scaler = train_scaler.fit(train_data[:,:8])'''

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

base_size = 100000 # for train and validation data each

fraction = 3*base_size/sample.events.shape[0] # fraction of the dataset that is actually needed

sample.events = sample.events.sample(frac=fraction, random_state=SEED, ignore_index=True)[-base_size:]

print(sample.events.shape)

zmasses = zcandidate.ZmassPairChooser(sample)
leptons = zmasses.find_Z()

print(sample.events.shape)

kin_variables = angles.calculate(leptons.T[0], leptons.T[1], leptons.T[2], leptons.T[3])

print(sample.events.shape)

c6_values = [-10]

c6_mod = trilinear.Modifier(c6_values = [-5,-1,0,1,5], c6_amplitudes = ['msq_sbi_c6_6', 'msq_sbi_c6_10', 'msq_sbi_c6_11', 'msq_sbi_c6_12', 'msq_sbi_c6_16'])
c6_weights = c6_mod.modify(sample=sample, c6=c6_values)

sig_weights = tf.convert_to_tensor(c6_weights)
sig_weights *= 1/tf.reduce_sum(sig_weights)

bkg_weights = tf.convert_to_tensor(sample.events['wt'])[:,tf.newaxis]
bkg_weights *= 1/tf.reduce_sum(bkg_weights)

test_data = tf.concat([tf.convert_to_tensor(kin_variables),sig_weights, bkg_weights], axis=1)

scaler = StandardScaler()

#scaler.mean_ = np.array([0.00019189256612400722, 0.6705156706726307, 0.672182766320336, -0.002330860289531429, 0.0006044017928423123, 91.30343432815204, 91.34168283098509, 258.4664139166857])
#scaler.scale_ = np.sqrt(np.array([0.7630219446604002, 0.3005563388031857, 0.29824529928165605, 3.2174175459094077, 3.0913094169204043, 27.71579453785725, 27.388630900483534, 6627.097114620871]))

scaler.mean_ = np.array([0.00019189256612400722, 0.6705156706726307, 0.672182766320336, -0.002330860289531429, 0.0006044017928423123, 91.30343432815204, 91.34168283098509, 258.4664139166857])
scaler.scale_ = np.array([0.87351127, 0.54823019, 0.54611839, 1.79371613, 1.75821199, 5.26457924, 5.23341484, 81.40698443])

#print('Mean:',train_scaler.mean_)
#print('Scale:',train_scaler.scale_)
#print('Variance:',train_scaler.var_)

test_data = tf.concat([scaler.transform(test_data[:,:8]), test_data[:,8:]], axis=1)

print(test_data, test_data.shape)

mu = tf.reduce_sum(test_data[:,:8], axis=0)/test_data.shape[0]

print('Mean (after):', mu)

sigma = tf.math.reduce_std(test_data[:,:8], axis=0)

print('sigma (after):', sigma)

model = keras.models.load_model(OUTPUT_DIR + f'/ckpt/checkpoint.model_{GEN}.tf', custom_objects={'C6_4l_clf_maxi_nonprm': C6_4l_clf_maxi_nonprm, 'swish_activation': swish_activation})


arr_len = test_data.shape[0]/len(c6_values)

data = test_data[:,:8][:,np.newaxis]
predictions = model.predict(data, verbose=2)

real_ratios = test_data[:,8]/test_data[:,9]

predictions = tf.convert_to_tensor(predictions)

ratios = tf.squeeze(predictions/(1-predictions), axis=2)

print(ratios, tf.math.reduce_min(ratios), tf.math.reduce_max(ratios))

real_ratios = tf.convert_to_tensor(real_ratios)

print(real_ratios, tf.math.reduce_min(real_ratios), tf.math.reduce_max(real_ratios))

real = real_ratios.numpy()
pred = ratios.numpy()

lnspc = np.linspace(0.6,1.4)

plt.figure(figsize=(6,6))

plt.plot(lnspc, lnspc, color='dimgray', linestyle=(0,(5,10)), label='NN estimate = true')
plt.scatter(real, pred, s=10, marker='x', label='Estimated data')
plt.xlabel(u'True ratio   $P(x|c_6)/P_0(x)$')
plt.ylabel(u'NN estimated   $P(x|c_6)/P_0(x)$')
plt.xlim(0.6,1.4)
plt.ylim(0.6,1.4)
plt.legend()

plt.savefig('pred_6.pdf')

plt.clf()

hist_prm = ''

with open(OUTPUT_DIR + f'/history/history_{GEN}.txt', 'r') as hist_file:
    hist_prm = hist_file.readlines()

hist_prm = [ np.array(el.replace('[','').replace(']','').replace(' ','').split(','), dtype=float) for el in hist_prm[0].split('][')]

t_loss_prm = np.array(hist_prm[0])
v_loss_prm = np.array(hist_prm[1])

epochs = range(1,t_loss_prm.shape[0]+1)

fig = plt.figure(figsize=(8,6))

#ax1.set_xticklabels([])

plt.plot(epochs, t_loss_prm, 'b', label='Training loss')
plt.xlabel('epochs []')
plt.ylabel('loss []')
#ax2.legend()

plt.plot(epochs, v_loss_prm, 'r', label='Validation loss')
#ax1.set_ylabel('loss []')
plt.legend()

fig.tight_layout()

plt.savefig('loss_6.pdf')

plt.clf()