import numpy as np
import pandas as pd

from .process import Basis

class Modifier():
  def __init__(self, component = Basis.SBI, c6_values = [-5,-1,0,1,5], c6_amplitudes = ['msq_sbi_c6_6', 'msq_sbi_c6_10', 'msq_sbi_c6_11', 'msq_sbi_c6_12', 'msq_sbi_c6_16']):
    self.component = component
    self.c6_values = np.array(c6_values)
    self.c6_amplitudes = list(c6_amplitudes)

  def modify(self, sample, c6):
    if np.isscalar(c6):
      c6 = [c6]

    msq_c6 = np.array([sample.events[c6_amplitude] for c6_amplitude in self.c6_amplitudes]).T
    msq_sm = np.array(sample.events[sample.components[self.component]])
    # Solve the polynomial for each row
    coeffs = np.apply_along_axis(lambda x: np.linalg.solve(np.vander(self.c6_values, len(self.c6_values), increasing=True), x), 1, msq_c6 / msq_sm[:, np.newaxis])[:, ::-1]
    # Evaluate the polynomial at c6 for each row
    return sample.reweight(self.component)[:,np.newaxis] * np.apply_along_axis(lambda x: np.polyval(x, np.array(c6)), 1, coeffs)