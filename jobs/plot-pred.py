import sys
sys.path.append('../')

import tensorflow as tf
from tensorflow import keras

from nn.models import C6_4l_clf_maxi, swish_activation
from nn import datasets
from hstar import gghzz, c6, msq
from hzz import zpair, angles

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

SEED=373485
GEN=4

EVENT_NUM = 10000

OUTPUT_DIR='../outputs/def'
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

c6_values = np.linspace(-20,20,21)

c6_mod = c6.Modifier(amplitude_component = msq.Component.SBI, c6_values = [-5,-1,0,1,5])
c6_weights, c6_prob = c6_mod.modify(sample=sample, c6=c6_values)

test_data = []

scaler = StandardScaler()
scaler.mean_ = 0.0 # change
scaler.scale_ = 0.0 # change

for i in range(len(c6_values)):
  sig_weights = tf.convert_to_tensor(c6_prob.T[i])[:,tf.newaxis]

  bkg_weights = tf.convert_to_tensor(sample.events.probabilities)[:,tf.newaxis]

  test_data_i = tf.concat([tf.convert_to_tensor(kin_variables),sig_weights, bkg_weights], axis=1)

  test_data_i = tf.concat([scaler.transform(test_data_i[:,:8]), test_data_i[:,8:]], axis=1)

  test_data.append(test_data_i)

test_data = tf.convert_to_tensor(test_data)

print(test_data.shape)

model = keras.models.load_model(OUTPUT_DIR + f'/ckpt/checkpoint.model_{GEN}.tf', custom_objects={'C6_4l_clf_maxi': C6_4l_clf_maxi, 'swish_activation': swish_activation})

arr_len = test_data.shape[0]/len(c6_values)

t_ratios = []

p_ratios = []

for i in range(len(c6_values)):
  c6 = c6_values[i]
  data = tf.concat([test_data[i,:1000,:8], tf.cast(tf.ones((1000,1))*c6, tf.float64)], axis=1)
  pred = model.predict(data, verbose=2)
  t_ratio = test_data[i,:1000,8]/test_data[i,:1000,9]
  p_ratio = tf.squeeze(pred/(1-pred))
  t_ratios.append(t_ratio)
  p_ratios.append(p_ratio)


t_ratios = tf.convert_to_tensor(t_ratios)
p_ratios = tf.convert_to_tensor(p_ratios)

print(t_ratios, tf.math.reduce_min(t_ratios), tf.math.reduce_max(t_ratios))
print(p_ratios, tf.math.reduce_min(p_ratios), tf.math.reduce_max(p_ratios))

real = t_ratios.numpy()
pred = p_ratios.numpy()

lnspc = np.linspace(0.0,2.0)

plt.figure(figsize=(10,10))

rand_int = int(np.round(np.random.rand()))

plt.plot(lnspc, lnspc, color='dimgray', linestyle=(0,(5,10)), label='NN estimate = true')
plt.scatter(real[rand_int], pred[rand_int], s=10, marker='x', label='Estimated data', color='orangered')
#plt.scatter(real[1], pred[1], s=10, marker='x', label='event #2', color='mistyrose')
#plt.scatter(real[2], pred[2], s=10, marker='x', label='event #3', color='orange')
#plt.scatter(real[3], pred[3], s=10, marker='x', label='event #4', color='olive')
#plt.scatter(real[4], pred[4], s=10, marker='x', label='event #5', color='yellow')
#plt.scatter(real[5], pred[5], s=10, marker='x', label='event #6', color='darkseagreen')
#plt.scatter(real[6], pred[6], s=10, marker='x', label='event #7', color='limegreen')
#plt.scatter(real[7], pred[7], s=10, marker='x', label='event #8', color='deepskyblue')
#plt.scatter(real[8], pred[8], s=10, marker='x', label='event #9', color='midnightblue')
#plt.scatter(real[9], pred[9], s=10, marker='x', label='event #10', color='darkviolet')
plt.xlabel(u'True ratio   $P(x|c_6)/P_0(x)$')
plt.ylabel(u'NN estimated   $P(x|c_6)/P_0(x)$')
plt.xlim(0.6,1.2)
plt.ylim(0.6,1.2)
plt.legend()

plt.savefig('pred_4.pdf')

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

plt.savefig('loss_4.pdf')

plt.clf()