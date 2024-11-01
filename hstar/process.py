import numpy as np
import pandas as pd
from enum import Enum

class Basis(Enum):
  SIG = 1
  INT = 2
  BKG = 3
  SBI = 4

class Channel(Enum):
  EL4 = 1
  MU4 = 2
  EL2MU2 = 3

class Sample():
  def __init__(self, weight='wt', amplitude = Basis.SBI, components = {
    Basis.SBI: 'msq_sbi_sm',
    Basis.SIG: 'msq_sig_sm',
    Basis.BKG: 'msq_bkg_sm',
    Basis.INT: 'msq_int_sm'
  }):
    self.weight = weight
    self.amplitude = amplitude
    self.components = components

  def open(self, csv, lumi, xs, *, k = 1.0):
    xs = np.array(xs) * k # times by k-factor
    # single-channel
    if np.isscalar(xs):
      assert isinstance(csv, str)
      self.events = pd.read_csv(csv)
      self.events[self.weight] *= xs * lumi / np.sum(self.events[self.weight])
    # normalize per-channel
    else:
      assert (isinstance(csv, list) and len(csv) == len(xs))
      events_per_channel = []
      for ichannel, filepath in enumerate(csv):
        events = pd.read_csv(filepath, nrows=10000)
        events[self.weight] *= (xs[ichannel] * lumi / np.sum(events[self.weight]))
        events_per_channel.append(events)
      self.events = pd.concat(events_per_channel)

  def reweight(self, component):
    return np.array(self.events[self.weight] * self.events[self.components[component]] / self.events[self.components[self.amplitude]])
