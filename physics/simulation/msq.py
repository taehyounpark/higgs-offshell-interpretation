from enum import Enum
import numpy as np

class Component(Enum):
  SBI = 4
  SIG = 1
  INT = 2
  BKG = 3

class ZeroMSQFilter():
  def __init__(self, amplitude):
    self.amplitude = amplitude

  def filter(self, events):
    indices = np.where(np.array(events.amplitudes[self.amplitude])!=0.0)[0]
    return indices, None