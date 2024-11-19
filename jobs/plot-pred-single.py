import sys
sys.path.append('../')

import tensorflow as tf
from tensorflow import keras

from nn.models import C6_4l_clf_maxi_nonprm, swish_activation
from nn import datasets
from hstar import gghzz, c6, msq
from hzz import zpair, angles

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

SEED=373485
GEN=7

EVENT_NUM = 100000

OUTPUT_DIR='../outputs/single'
SAMPLE_DIR='../..'

sample = gghzz.Process(  
    (1.4783394, SAMPLE_DIR + '/ggZZ2e2m_all_new.csv', 1e6),
    (0.47412769, SAMPLE_DIR + '/ggZZ4e_all_new.csv', 1e6),
    (0.47412769, SAMPLE_DIR + '/ggZZ4m_all_new.csv', 1e6)
)

base_size = EVENT_NUM # for train and validation data each

fraction = 3*base_size/sample.events.shape[0] # fraction of the dataset that is actually needed

sample.events = sample.events.sample(frac=fraction, random_state=SEED)[-base_size:]

z_chooser = zpair.ZPairChooser(bounds1=(50,115), bounds2=(50,115), algorithm='leastsquare')
l1_1, l2_1, l1_2, l2_2 = sample.events.filter(z_chooser)

kin_variables = angles.calculate(l1_1, l2_1, l1_2, l2_2)

c6_values = [-10]

c6_mod = c6.Modifier(amplitude_component = msq.Component.SBI, c6_values = [-5,-1,0,1,5])
c6_weights, c6_prob = c6_mod.modify(sample=sample, c6=c6_values)

sig_weights = tf.convert_to_tensor(c6_prob)

bkg_weights = tf.convert_to_tensor(sample.events.probabilities)

test_data = tf.concat([tf.convert_to_tensor(kin_variables),sig_weights, bkg_weights], axis=1)

scaler = StandardScaler()
test_data = tf.concat([scaler.fit_transform(test_data[:,:8]), test_data[:,8:]], axis=1)

print(test_data, test_data.shape)

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

lnspc = np.linspace(0.0,2.0)

plt.figure(figsize=(6,6))

plt.plot(lnspc, lnspc, color='dimgray', linestyle=(0,(5,10)), label='NN estimate = true')
plt.scatter(real, pred, s=10, marker='x', label='Estimated data')
plt.xlabel(u'True ratio   $P(x|c_6)/P_0(x)$')
plt.ylabel(u'NN estimated   $P(x|c_6)/P_0(x)$')
plt.xlim(0.6,1.4)
plt.ylim(0.6,1.4)
plt.legend()

plt.savefig('pred_5.pdf')

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

plt.savefig('loss_5.pdf')

plt.clf()