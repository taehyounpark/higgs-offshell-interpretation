import pandas as pd

from ..simulation import mcfm

class Events():
  def __init__(self):
    self.kinematics = None
    self.amplitudes = None
    self.weights = None
    self.probabilities = None

  def filter(self, obj_instance):
    indices, output = obj_instance.filter(self)

    self.kinematics = self.kinematics.take(indices)
    self.amplitudes = self.amplitudes.take(indices)
    self.weights = self.weights.take(indices)
    self.probabilities = self.weights/self.weights.sum()

    return output

  def shuffle(self, random_state=None):
    events = Events()

    events.kinematics = self.kinematics.sample(frac=1.0, random_state=random_state, ignore_index=True)
    events.amplitudes = self.amplitudes.sample(frac=1.0, random_state=random_state, ignore_index=True)
    events.weights = self.weights.sample(frac=1.0, random_state=random_state, ignore_index=True)
    events.probabilities = events.weights/events.weights.sum()

    return events
  
  def sample(self, frac=1.0, random_state=None):
    events = Events()

    events.kinematics = self.kinematics.sample(frac=frac, random_state=random_state, ignore_index=True)
    events.amplitudes = self.amplitudes.sample(frac=frac, random_state=random_state, ignore_index=True)
    events.weights = self.weights.sample(frac=frac, random_state=random_state, ignore_index=True)
    events.probabilities = events.weights/events.weights.sum()

    return events

  def __getitem__(self, item):
    events = Events()
    
    events.kinematics = self.kinematics[item]
    events.amplitudes = self.amplitudes[item]
    events.weights = self.weights[item]
    events.probabilities = events.weights/events.weights.sum()

    return events

class Process():

  def __init__(self, *channels):

    self.events = Events()

    kinematics_per_channel = []
    amplitudes_per_channel = []
    weights_per_channel = []

    for sample_from_channel in channels:
      xsec = sample_from_channel[0]
      filepath = sample_from_channel[1]
      nrows = None if len(sample_from_channel) < 3 else sample_from_channel[2]
      df = pd.read_csv(filepath, nrows=nrows)
      kinematics_per_channel.append(df[mcfm.kinematics])
      amplitudes_per_channel.append(df[mcfm.amplitudes])
      weights = df[mcfm.weight]
      # normalize
      weights *= xsec / weights.sum() 
      weights_per_channel.append(weights)

    self.events.kinematics = pd.concat(kinematics_per_channel)
    self.events.amplitudes = pd.concat(amplitudes_per_channel)
    self.events.weights = pd.concat(weights_per_channel)
    self.events.probabilities = self.events.weights/self.events.weights.sum()

  def __getitem__(self, component):
    events = Events()
    events.kinematics = self.events.kinematics
    events.amplitudes = self.events.amplitudes
    events.weights = self.events.weights * events.amplitudes[mcfm.amplitude_sm[component]] / events.amplitudes[mcfm.amplitude_base]
    events.probabilities = events.weights / events.weights.sum()
    return events