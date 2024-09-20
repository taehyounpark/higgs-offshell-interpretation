import numpy as np

def nll(obs, exp):
  from scipy.special import loggamma
  condition = (exp!=0)
  obs = obs[condition]
  exp = exp[condition]
  return np.sum( - ( obs * np.log(exp) - exp - loggamma(obs+1) ) )