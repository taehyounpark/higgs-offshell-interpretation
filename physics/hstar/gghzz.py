import pandas as pd

from ..simulation import mcfm

class Events():
  def __init__(self):
    self.kinematics = None
    self.components = None
    self.weights = None
    self.probabilities = None

  def filter(self, obj_instance):
    indices, output = obj_instance.filter(self)

    self.kinematics = self.kinematics.take(indices)
    self.components = self.components.take(indices)
    self.weights = self.weights.take(indices)
    self.probabilities = self.weights/self.weights.sum()

    return output

  def shuffle(self, random_state=None):
    events = Events()

    events.kinematics = self.kinematics.sample(frac=1.0, random_state=random_state, ignore_index=True)
    events.components = self.components.sample(frac=1.0, random_state=random_state, ignore_index=True)
    events.weights = self.weights.sample(frac=1.0, random_state=random_state, ignore_index=True)
    events.probabilities = events.weights/events.weights.sum()

    return events
  
  def sample(self, frac=1.0, random_state=None):
    events = Events()

    events.kinematics = self.kinematics.sample(frac=frac, random_state=random_state, ignore_index=True)
    events.components = self.components.sample(frac=frac, random_state=random_state, ignore_index=True)
    events.weights = self.weights.sample(frac=frac, random_state=random_state, ignore_index=True)
    events.probabilities = events.weights/events.weights.sum()

    return events

  def __getitem__(self, item):
    events = Events()
    
    events.kinematics = self.kinematics[item]
    events.components = self.components[item]
    events.weights = self.weights[item]
    events.probabilities = events.weights/events.weights.sum()

    return events

class Process():

  def __init__(self, baseline, *channels):
    self.baseline = baseline
    self.events = Events()

    kinematics_per_channel = []
    components_per_channel = []
    weights_per_channel = []

    for sample_from_channel in channels:
      xsec = sample_from_channel[0]
      
      if not isinstance(sample_from_channel[1],pd.DataFrame):
        filepath = sample_from_channel[1]
        nrows = None if len(sample_from_channel) < 3 else sample_from_channel[2]
        df = pd.read_csv(filepath, nrows=nrows)
      else:
        df = sample_from_channel[1]
      kinematics_per_channel.append(df[mcfm.kinematics])
      components_per_channel.append(df[mcfm.components])
      weights = df[mcfm.weight]
      # normalize
      weights *= xsec / weights.sum() 
      weights_per_channel.append(weights)

    self.events.kinematics = pd.concat(kinematics_per_channel)
    self.events.components = pd.concat(components_per_channel)
    self.events.weights = pd.concat(weights_per_channel)
    self.events.probabilities = self.events.weights/self.events.weights.sum()

  def __getitem__(self, component):
    events = Events()
    events.kinematics = self.events.kinematics
    events.components = self.events.components
    events.weights = self.events.weights * events.components[mcfm.component_sm[component]] / events.components[mcfm.component_sm[self.baseline]]
    events.probabilities = events.weights / events.weights.sum()
    return events