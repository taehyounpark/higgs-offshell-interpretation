import numpy as np

from . import mcfm, msq

class Modifier():

  def __init__(self, amplitude_component = msq.Component.SBI, c6_values = [-5,-1,0,1,5]):
    self.amplitude_component = amplitude_component
    self.c6_values = np.array(c6_values)
    self.c6_amplitudes = [mcfm.amplitude_c6[amplitude_component][c6_value] for c6_value in c6_values]

  def modify(self, sample, c6):

    if np.isscalar(c6):
      c6 = [c6]

    events = sample[self.amplitude_component]

    msq_c6 = np.array([events.amplitudes[c6_amplitude].to_numpy() for c6_amplitude in self.c6_amplitudes]).T
    msq_sm = events.amplitudes[mcfm.amplitude_sm[self.amplitude_component]].to_numpy()

    # Solve the polynomial for each row
    coeffs = np.apply_along_axis(lambda x: np.linalg.solve(np.vander(self.c6_values, len(self.c6_values), increasing=True), x), 1, msq_c6 / msq_sm[:, np.newaxis])[:, ::-1]

    # Evaluate the polynomial at c6 for each row
    wt_c6 = events.weights.to_numpy()[:,np.newaxis] * np.apply_along_axis(lambda x: np.polyval(x, np.array(c6)), 1, coeffs)
    prob_c6 = wt_c6 / np.sum(wt_c6, axis=0)

    return (wt_c6, prob_c6)

      # cH_modification = -1.0 * events.weights[:,np.newaxis] * np.array(cH)

      # c6_modification = c6_modification[:, np.newaxis, :]

      # cH_modification = cH_modification[:, :, np.newaxis]

      # tot_modification = c6_modification + cH_modification 