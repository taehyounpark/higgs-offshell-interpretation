import numpy as np
from .sample import weight_key, reweight

reweight_map = {
  'rwt_6': -5,
  'rwt_10': -1,
  'rwt_11': 0,
  'rwt_12': 1,
  'rwt_16': 5
} 

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

def morph_event(event, c6):
  c6_reweights = np.array([event[c6_reweight_key] for c6_reweight_key in reweight_map.keys()])
  c6_values = list(reweight_map.values())
  event_reweighting = solve_polynomial(c6_values, c6_reweights)
  return event[weight_key] * np.polyval(event_reweighting, c6)

def morph_events(events, c6):
  morphed_events = []
  for _,event in events.iterrows():
    morphed_events.append(morph_event(event, c6))
  return np.array(morphed_events)

def morph_yield(events, c6):
  sm_yield = np.sum(events[weight_key])
  c6_yields = [reweight(events, c6_reweight_key) for c6_reweight_key in reweight_map.keys()]
  c6_values = list(reweight_map.values())
  yield_reweighting = solve_polynomial(c6_values, c6_yields / sm_yield)
  return sm_yield * np.polyval(yield_reweighting, c6)

def morph(event_or_events, c6, *, per_event=True):
  if event_or_events.ndim == 1:
    return morph_event(event_or_events, c6)
  elif event_or_events.ndim == 2:
    if per_event:
      return morph_events(event_or_events, c6)
    else:
      return morph_yield(event_or_events, c6)
