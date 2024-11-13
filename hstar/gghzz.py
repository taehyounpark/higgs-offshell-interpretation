import numpy as np
import pandas as pd
from enum import Enum

from . import mcfm, amplitude

class Channel(Enum):
  ELEL = 1
  MUMU = 2
  ELMU = 3
  INCL = 4

class Events():
  def __init__(self):
    self.kinematics = None
    self.amplitudes = None
    self.weights = None
    self.probabilities = None

class Process():

  def __init__(self, *channels):

    self.events = Events()

    kinematics_per_channel = []
    amplitudes_per_channel = []
    weights_per_channel = []
    probabilities_per_channel = []

    for sample_from_channel in channels:
      xsec = sample_from_channel[0]
      filepath = sample_from_channel[1]
      nrows = None if len(sample_from_channel) < 2 else sample_from_channel[2]
      df = pd.read_csv(filepath, nrows=nrows)
      kinematics_per_channel.append(df[mcfm.kinematics])
      amplitudes_per_channel.append(df[mcfm.amplitudes])
      weights = df[mcfm.weight]
      # normalize
      weights *= xsec / np.sum(weights) 
      weights_per_channel.append(weights)
      probabilities_per_channel.append(weights / xsec)

    self.events.kinematics = pd.concat(kinematics_per_channel)
    self.events.amplitudes = pd.concat(amplitudes_per_channel)
    self.events.weights = pd.concat(weights_per_channel)
    self.events.probailities = pd.concat(probabilities_per_channel)

  def __getitem__(self, component):
    events = Events()
    events.kinematics = self.events.kinematics
    events.amplitudes = self.events.amplitudes
    events.weights = self.events.weights * events.amplitudes[mcfm.amplitude_sm[component]] / events.amplitudes[mcfm.amplitude_base]
    events.probabilities = events.weights / np.sum(events.weights)
    return events