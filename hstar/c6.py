import numpy as np

def solve_polynomial(x, y):
  """
  Finds the coefficients of the polynomial that fits the given (x, y) coordinates.

  Parameters:
  x (array-like): An array of x coordinates.
  y (array-like): An array of y coordinates.

  Returns:
  numpy.ndarray: The coefficients of the polynomial.
  """
  x = np.array(x)
  y = np.array(y)
  N = len(x)
  
  # construct the Vandermonde matrix
  V = np.vander(x, N, increasing=True)
  
  # solve for the polynomial coefficients
  coeffs = np.linalg.solve(V, y)

  # return the function
  return coeffs[::-1]

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

  def _morph_msq_one_event(self, event, c6):
    msq_sm = event[self.sm_msq_key]
    msq_c6 = np.array([event[c6_msq_key] for c6_msq_key in self.c6_msq_map.values()])
    c6_vals = list(self.c6_msq_map.keys())
    coeffs = solve_polynomial(c6_vals, msq_c6 / msq_sm)
    return np.polyval(coeffs, c6)

  def _morph_msq_per_event(self, c6):
    if np.isscalar(c6):
      msq_morphing = np.ones_like(self.events[self.sm_msq_key])
      for ievent, event in self.events.iterrows():
        msq_morphing[ievent] = self._morph_msq_one_event(event, c6)
      return np.array(msq_morphing)
    else:
      msq_morphing = np.ones((len(self.events[self.sm_msq_key]), len(c6)))
      for ievent, event in self.events.iterrows():
        msq_morphing[ievent, :] = self._morph_msq_one_event(event, c6)
      return np.array(msq_morphing)
      
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