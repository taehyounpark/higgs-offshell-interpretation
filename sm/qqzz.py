import pandas as pd

class Process():
  def __init__(self, *channels):
    self.kinematics = None
    self.amplitudes = None
    self.weights = None
    self.probabilities = None