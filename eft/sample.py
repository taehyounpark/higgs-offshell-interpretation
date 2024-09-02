import numpy as np

weight_key = 'wt'

def normalize(events, xsec : float, lumi : float):
  sumw = np.sum(events[weight_key])
  events[weight_key] *= xsec * lumi / sumw
  return events

def reweight(events, reweight_key = ''):
  return np.sum(events[weight_key] * events[reweight_key])