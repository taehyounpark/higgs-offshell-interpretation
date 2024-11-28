import sys
sys.path.append('../')

import vector
import numpy as np

import pandas as pd

from hstar import gghzz

class ZPairChooser:
    def __init__(self, bounds1: tuple[int, int], bounds2: tuple[int, int], algorithm: str = 'leastsquare'):
        self.bounds1 = bounds1
        self.bounds2 = bounds2

        if algorithm not in ['leastsquare', 'closest']:
            raise ValueError('algorithm has to be one of ["leastsquare", "closest"]')

        self.algorithm = algorithm

        self.Z_mass = 91.18
    
    def find_Z_lsq(self, l1, l2, l3, l4):
        # Possible Z bosons from leptons 
        p12 = (l1 + l2)
        p14 = (l1 + l4)
        p23 = (l2 + l3)
        p34 = (l3 + l4)

        # Possible Z boson pairs as Momentum4D objects in vector arrays
        pairs = vector.array([[p12, p34], [p14, p23]], dtype=[('px',float),('py',float),('pz',float),('E',float)])
        lepton_pairs = vector.array([[[l1,l2],[l3,l4]],
                                      [[l1,l4],[l3,l2]]], dtype=[('px',float),('py',float),('pz',float),('E',float)])

        # Squared minimization to determine the closest pair
        sq = np.array([(pair[0].mass - self.Z_mass)**2 + (pair[1].mass - self.Z_mass)**2 for pair in pairs]).T
        closest_pair_indices = np.argmin(sq, axis=1)
        closest_pair = pairs.transpose(2,0,1)[np.arange(len(closest_pair_indices)), closest_pair_indices].T

        # Determine the Z boson with the higher pT
        # That one will be Z1, the other one Z2
        pT_max_ind = np.argmax(closest_pair.pt,axis=0) # Z1
        pT_min_ind = np.argmin(closest_pair.pt,axis=0) # Z2

        # Determine the order manually if both Z bosons have the same pT
        cond=(pT_max_ind==pT_min_ind)

        pT_max_ind[cond] = 0
        pT_min_ind[cond] = 1

        # (l1_1, l2_1) = Z1; (l1_2, l2_2) = Z2
        l1_1, l2_1 = lepton_pairs.transpose(3,0,1,2)[np.arange(len(closest_pair_indices)), closest_pair_indices][np.arange(len(pT_max_ind)), pT_max_ind].T
        l1_2, l2_2 = lepton_pairs.transpose(3,0,1,2)[np.arange(len(closest_pair_indices)), closest_pair_indices][np.arange(len(pT_min_ind)), pT_min_ind].T

        return (l1_1, l2_1, l1_2, l2_2)
    
    def find_Z_closest(self, l1, l2, l3, l4):
        # Possible Z bosons from leptons 
        p12 = (l1 + l2)
        p14 = (l1 + l4)
        p23 = (l2 + l3)
        p34 = (l3 + l4)

        # Possible Z boson pairs as Momentum4D objects in vector arrays
        pairs = vector.array([[p12, p34], [p14, p23]], dtype=[('px',float),('py',float),('pz',float),('E',float)])
        lepton_pairs = vector.array([[[l1,l2],[l3,l4]],
                                      [[l1,l4],[l3,l2]]], dtype=[('px',float),('py',float),('pz',float),('E',float)])

        # Just choose the Z boson pair which contains the Z boson closest to the true rest mass
        pairs_diffs = ((pairs.mass - np.ones(pairs.shape)*self.Z_mass)**2).transpose(2,0,1).reshape(pairs.shape[2],4)
        min_ind = np.floor(np.argmin(pairs_diffs, axis=1)/2.0).astype(int)
        closest_Z_pair = pairs.transpose(2,0,1)[np.arange(len(min_ind)),min_ind].T

        closest_Z_min_ind = np.argmin((closest_Z_pair.mass-self.Z_mass)**2, axis=0)
        closest_Z_max_ind = np.argmax((closest_Z_pair.mass-self.Z_mass)**2, axis=0)

        # (l1_1, l2_1) = Z1; (l1_2, l2_2) = Z2
        l1_1, l2_1 = lepton_pairs.transpose(3,0,1,2)[np.arange(len(min_ind)), min_ind][np.arange(len(closest_Z_min_ind)), closest_Z_min_ind].T
        l1_2, l2_2 = lepton_pairs.transpose(3,0,1,2)[np.arange(len(min_ind)), min_ind][np.arange(len(closest_Z_max_ind)), closest_Z_max_ind].T

        return (l1_1, l2_1, l1_2, l2_2)

    def filter(self, events: gghzz.Events) -> tuple[np.ndarray, tuple[vector.MomentumNumpy4D, vector.MomentumNumpy4D, vector.MomentumNumpy4D, vector.MomentumNumpy4D]]:
        #Outgoing leptons
        l1 = vector.array({'px': events.kinematics['p3_px'], 'py': events.kinematics['p3_py'], 'pz': events.kinematics['p3_pz'], 'E': events.kinematics['p3_E']})#negative l1
        l2 = vector.array({'px': events.kinematics['p4_px'], 'py': events.kinematics['p4_py'], 'pz': events.kinematics['p4_pz'], 'E': events.kinematics['p4_E']})#positive l1
        l3 = vector.array({'px': events.kinematics['p5_px'], 'py': events.kinematics['p5_py'], 'pz': events.kinematics['p5_pz'], 'E': events.kinematics['p5_E']})#negative l2
        l4 = vector.array({'px': events.kinematics['p6_px'], 'py': events.kinematics['p6_py'], 'pz': events.kinematics['p6_pz'], 'E': events.kinematics['p6_E']})#positive l2

        if self.algorithm == 'leastsquare':
            l1_1, l2_1, l1_2, l2_2 = self.find_Z_lsq(l1, l2, l3, l4)
        elif self.algorithm == 'closest':
            l1_1, l2_1, l1_2, l2_2 = self.find_Z_closest(l1, l2, l3, l4)

        Z1 = l1_1 + l2_1
        Z2 = l1_2 + l2_2

        cond1 = np.where((Z1.mass>=self.bounds1[0])&(Z1.mass<=self.bounds1[1]))
        cond2 = np.where((Z2.mass>=self.bounds2[0])&(Z2.mass<=self.bounds2[1]))

        # Get only indices where cond1 and cond2 apply
        indices = np.intersect1d(cond1,cond2)

        l1_1 = l1_1[indices]
        l2_1 = l2_1[indices]
        l1_2 = l1_2[indices]
        l2_2 = l2_2[indices]

        return (indices, (l1_1, l2_1, l1_2, l2_2))