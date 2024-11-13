import numpy as np

def ratio(wts_a, wts_b):
  
  nexp_a = np.sum(wts_a, axis=0)
  nexp_b = np.sum(wts_b)
  nobs = nexp_b

  prob_a = wts_a / nexp_a
  prob_b = wts_b / nexp_b
  pratio_ab = prob_a / prob_b[:,np.newaxis,np.newaxis]
  
  return -2*nobs*(np.log(nexp_a) - np.log(nexp_b)) + 2*(nexp_a - nexp_b) - 2*np.sum( wts_b[:,np.newaxis,np.newaxis] * np.log(pratio_ab), axis=0)

class Scanner():

  def __init__(self, x, y, z):
    self._x = x
    self._y = y
    self._z = z

  def val(self):
    xi, yi = np.unravel_index(np.argmin(self._z), self._z.shape)
    return self._x[xi], self._y[yi]

  def ci(self, nsigma=1):
    # Step 1: Create a binary mask where values are less than Z_max
    z_max = nsigma**2
    mask = self._z < z_max
    
    # Step 2: Compute gradients to detect boundaries
    grad_x, grad_y = np.gradient(mask.astype(float))
    
    # Step 3: Find boundary coordinates
    boundary_indices = np.argwhere((np.abs(grad_x) > 0) | (np.abs(grad_y) > 0))

    # print(boundary_indices)

    # Plot the boundary coordinates as a smooth contour line
    x_coords = self._x[boundary_indices[:, 1]]
    y_coords = self._y[boundary_indices[:, 0]]
    
    return x_coords, y_coords
 