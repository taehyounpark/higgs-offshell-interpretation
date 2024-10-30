import numpy as np
import vector
from vector import MomentumObject4D
import tensorflow as tf


def calculate(l1: MomentumObject4D, l2: MomentumObject4D, l3: MomentumObject4D, l4: MomentumObject4D, tensorize: bool=False) -> np.ndarray:
    """
    Calculates kinematic variables cos 𝜃∗, cos 𝜃1, cos 𝜃2, 𝜙1 ,𝜙, mZ1 and mZ2. 
    Angles used are described in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.86.095031.

    params:
        l1 (MomentumObject4D): Lepton 1 used in calculations. Must be positive and paired up with l2 to form a Z boson. Can also be a vector array of MomentumObject4D objects.
        l2 (MomentumObject4D): Lepton 2 used in calculations. Must be negative and paired up with l1 to form a Z boson. Can also be a vector array of MomentumObject4D objects.
        l3 (MomentumObject4D): Lepton 3 used in calculations. Must be positive and paired up with l4 to form a Z boson. Can also be a vector array of MomentumObject4D objects.
        l4 (MomentumObject4D): Lepton 4 used in calculations. Must be negative and paired up with l3 to form a Z boson. Can also be a vector array of MomentumObject4D objects.
        tensorize (bool): If True, the output will be converted to a tensorflow tensor. False by default.

    Returns:
        np.ndarray: The kinematic variables in form of [cos 𝜃∗, cos 𝜃1, cos 𝜃2, 𝜙1 ,𝜙, mZ1, mZ2].
    """
    Z1 = l1+l2
    Z2 = l3+l4

    H = Z1+Z2

    # Boost both Z bosons to Higgs frame
    Z1_h = Z1.boost(-H)
    Z2_h = Z2.boost(-H)

    # Unit vector (3D) from boosted Z1 boson
    z1 = Z1_h.to_3D().unit()

    # Angle between Z1 and the beam axis
    cth_star = z1.z

    # Boost leptons to Higgs frame
    l1_h = l1.boost(-H)
    l2_h = l2.boost(-H)
    l3_h = l3.boost(-H)
    l4_h = l4.boost(-H)

    # Get 3D vectors
    l1_h3 = l1_h.to_3D()
    l2_h3 = l2_h.to_3D()
    l3_h3 = l3_h.to_3D()
    l4_h3 = l4_h.to_3D()

    # Get unit vector in z direction
    nz = vector.array({'x': np.zeros(Z1_h.shape[0]), 'y': np.zeros(Z1_h.shape[0]), 'z': np.ones(Z1_h.shape[0])})

    n12 = l1_h3.cross(l2_h3).unit() # Normal vector of the plane in which the Z1 decay takes place
    n34 = l3_h3.cross(l4_h3).unit() # Normal vector of the plane in which the Z2 decay takes place
    nscp = nz.cross(z1).unit() # Normal vector of the plane in which the H -> Z1, Z2 takes place

    # Calculate 𝜙1 ,𝜙
    phi = z1.dot(n12.cross(n34))/np.abs(z1.dot(n12.cross(n34)))*np.arccos(-n12.dot(n34))

    # TODO: decide what to do if n12 and n34 are parallel
        
    phi1 = z1.dot(n12.cross(nscp))/np.abs(z1.dot(n12.cross(nscp)))*np.arccos(n12.dot(nscp))

    # Boost Z1 to Z2 rest frame and Z2 to Z1 rest frame (does this really work like this?)
    Z1_in_Z2 = Z1_h.boost(-Z2_h)
    Z2_in_Z1 = Z2_h.boost(-Z1_h)

    z1_in_Z2 = Z1_in_Z2.to_3D()
    z2_in_Z1 = Z2_in_Z1.to_3D()

    l1 = l1.boost(-Z1_h)
    l3 = l3.boost(-Z2_h)

    # Calculate cos 𝜃1, cos 𝜃2
    cth1 = - z2_in_Z1.dot(l1.to_3D())/np.abs(z2_in_Z1.mag*l1.to_3D().mag)
    cth2 = - z1_in_Z2.dot(l3.to_3D())/np.abs(z1_in_Z2.mag*l3.to_3D().mag)

    if tensorize:
        return tf.convert_to_tensor(np.array([cth_star, cth1, cth2, phi1, phi, Z1.mass, Z2.mass]).T)
    else:
        return np.array([cth_star, cth1, cth2, phi1, phi, Z1.mass, Z2.mass]).T