import numpy as np
import pandas as pd

class Sample():

  def __init__(self, xs, events, *, k=1.0):
    self.sm_xs = xs * k
    self.sm_wt_key = 'wt'
    self.sm_msq_key = 'msq_sm'
    self.c6_msq_map = {
      -5 : 'msq_c6_6',
      -1 : 'msq_c6_10',
      0 : 'msq_c6_11',
      1 : 'msq_c6_12',
      5 : 'msq_c6_16'
    }
    self.events = events

  def normalize(self, lumi):
    self.events[self.sm_wt_key] *= self.sm_xs * lumi / np.sum(self.events[self.sm_wt_key])
    return self.events

  def _morph_msq_per_event(self, c6):
    c6_vals = np.array(list(self.c6_msq_map.keys()))
    msq_c6 = np.array([self.events[c6_msq_key] for c6_msq_key in self.c6_msq_map.values()]).T
    msq_sm = np.array(self.events[self.sm_msq_key])
    
    # Solve the polynomial for each row
    coeffs = np.apply_along_axis(lambda x: np.linalg.solve(np.vander(c6_vals, len(c6_vals), increasing=True), x), 1, msq_c6 / msq_sm[:, np.newaxis])[:, ::-1]
    
    # Evaluate the polynomial at c6 for each row
    return np.array([np.polyval(coeffs[i], c6) for i in range(len(coeffs))])
      
  def msq(self, c6=None):
    """
    Returns the matrix element (squared) for a given set of events.

    Parameters:
      c6 (float or array-like, optional): The value(s) of the Wilson coefficient c6.
        If None, returns the Standard Model value. If a scalar, returns the matrix element
        morphed, i.e. inter-/extra-polated, to the given C6 value. If an array, returns the value morphed to
        each value in the array.

    Returns:
      numpy.ndarray: The matrix element value(s) for the given events under the specified c6 value(s).
    """
    if c6 is None:
      return np.array(self.events[self.sm_msq_key])
    elif np.isscalar(c6):
      return self._morph_msq_per_event(c6) * np.array(self.events[self.sm_msq_key])
    else:
      return self._morph_msq_per_event(c6) * np.array(self.events[self.sm_msq_key])[:, np.newaxis]

  def nu(self, c6=None, per_event=False):
    if c6 is None:
      return np.array(self.events[self.sm_wt_key]) if per_event else np.sum(self.events[self.sm_wt_key])
    elif np.isscalar(c6):
      return np.array(self.events[self.sm_wt_key] * self._morph_msq_per_event(c6)) if per_event else np.sum(self.events[self.sm_wt_key] * self._morph_msq_per_event(c6))
    else:
      return np.array(self.events[self.sm_wt_key])[:, np.newaxis] * self._morph_msq_per_event(c6) if per_event else np.sum(np.array(self.events[self.sm_wt_key])[:, np.newaxis] * self._morph_msq_per_event(c6), axis=0)