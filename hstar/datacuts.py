import numpy as np
import vector
from vector import MomentumObject4D
import tensorflow as tf

from .process import Sample

from hzz.angles import calculate

class Events4l:
    def __init__(self, sample: Sample) -> None:
        """
        Initialize the particle momenta using a provided dataframe with columns 
        p1_x | p1_py | p1_pz | p1_E | p2_px | ... | p6_E 

        Here p1,p2 are the gluon momenta and p3,p4 are one lepton pair (l-,l+) while p5,p6 are the second lepton pair (l'-,l'+) (known only in simulated case)

        Additionally find most likely Z bosons from leptons and filter the data according to mass cuts.

        All momenta are described by using Momentum4D vector arrays.

        params: 
            df (DataFrame): Dataframe mentioned above 
        """
        self.df = sample.events
        self.sample = sample

        #Constants
        self.Z_mass = 91.18

        #Incoming gluons
        self.g1 = vector.array({'px': self.df['p1_px'], 'py': self.df['p1_py'], 'pz': self.df['p1_pz'], 'E': self.df['p1_E']})
        self.g2 = vector.array({'px': self.df['p2_px'], 'py': self.df['p2_py'], 'pz': self.df['p2_pz'], 'E': self.df['p2_E']})

        #Outgoing leptons
        self.l1 = vector.array({'px': self.df['p3_px'], 'py': self.df['p3_py'], 'pz': self.df['p3_pz'], 'E': self.df['p3_E']})#negative l1
        self.l2 = vector.array({'px': self.df['p4_px'], 'py': self.df['p4_py'], 'pz': self.df['p4_pz'], 'E': self.df['p4_E']})#positive l1
        self.l3 = vector.array({'px': self.df['p5_px'], 'py': self.df['p5_py'], 'pz': self.df['p5_pz'], 'E': self.df['p5_E']})#negative l2
        self.l4 = vector.array({'px': self.df['p6_px'], 'py': self.df['p6_py'], 'pz': self.df['p6_pz'], 'E': self.df['p6_E']})#positive l2

        self.m4l = (self.l1 + self.l2 + self.l3 + self.l4).mass

        #print('initialized momenta')

        #Find Z bosons from leptons
        self.find_Z()

        #print('found Z')

        self.filter_Z()

        #print('filtered events')


    def find_Z_alt(self) -> None:
        p12 = (self.l1 + self.l2)
        p14 = (self.l1 + self.l4)
        p23 = (self.l2 + self.l3)
        p34 = (self.l3 + self.l4)

        

        

    def find_Z(self) -> None:
        """
        Find the lepton pairs whose invariant masses are closest to the true Z boson rest mass by using chi squared minimization.
        """
        # Possible Z bosons from leptons 
        p12 = (self.l1 + self.l2)
        p14 = (self.l1 + self.l4)
        p23 = (self.l2 + self.l3)
        p34 = (self.l3 + self.l4)

        # Possible Z boson pairs as Momentum4D objects in vector arrays
        pairs = vector.array([[p12, p34], [p14, p23]], dtype=[('px',float),('py',float),('pz',float),('E',float)])
        lepton_pairs = vector.array([[[self.l2,self.l1],[self.l4,self.l3]],
                                      [[self.l4,self.l1],[self.l2,self.l3]]], dtype=[('px',float),('py',float),('pz',float),('E',float)])

        # Chi squared minimization to determine the closest pair
        chi_sq = np.array([(pair[0].mass - self.Z_mass)**2 + (pair[1].mass - self.Z_mass)**2 for pair in pairs]).T
        closest_pair_indices = np.argmin(chi_sq, axis=1)
        closest_pair = pairs.transpose(2,0,1)[np.arange(len(closest_pair_indices)), closest_pair_indices].T

        # Determine the Z boson with the higher pT
        # That one will be Z1, the other one Z2
        pT_min_ind = np.argmin(closest_pair.pt,axis=0) # Z1
        pT_max_ind = np.argmax(closest_pair.pt,axis=0) # Z2

        # The simulation uses Z1=(l1,l2), Z2=(l3,l4)
        # Get the true pair for comparison
        true_ind = np.zeros(chi_sq.shape[0], dtype=int) #   we know this from the simulation
        true_pair = pairs.transpose(2,0,1)[np.arange(len(true_ind)), true_ind].T

        # Old method kept for comparisons 
        # Just choose the Z boson pair which contains the Z boson closest to the true rest mass
        pairs_diffs = ((pairs.mass - np.ones(pairs.shape)*self.Z_mass)**2).transpose(2,0,1).reshape(pairs.shape[2],4)
        min_ind = np.floor(np.argmin(pairs_diffs, axis=1)/2.0).astype(int)
        closest_Z_pair = pairs.transpose(2,0,1)[np.arange(len(min_ind)),min_ind].T

        closest_Z_min_ind = np.argmin((closest_pair.mass-self.Z_mass)**2, axis=0)
        closest_Z_max_ind = np.argmax((closest_pair.mass-self.Z_mass)**2, axis=0)
        
        # Set variables for comparing methods
        self.pairs = pairs
        self.closest_pair_chi_sq = closest_pair
        self.closest_pair_lim = closest_Z_pair
        self.true_pair = true_pair

        #print('Z1 masses different in ',np.sum((~(closest_pair.T[np.arange(len(pT_min_ind)),pT_min_ind].mass==closest_Z_pair.T[np.arange(len(pT_min_ind)),pT_min_ind].mass)).astype(int))/closest_pair.shape[1]*100, '% of cases')
        #print('Z2 masses different in ',np.sum((~(closest_pair.T[np.arange(len(pT_max_ind)),pT_max_ind].mass==closest_Z_pair.T[np.arange(len(pT_max_ind)),pT_max_ind].mass)).astype(int))/closest_pair.shape[1]*100, '% of cases')

        # Set Z1, Z2 and (l1_calc, l2_calc) = Z1; (l3_calc, l4_calc) = Z2
        self.Z1 = closest_pair.T[np.arange(len(pT_max_ind)),pT_max_ind]
        self.l1_calc, self.l2_calc = lepton_pairs.transpose(3,0,1,2)[np.arange(len(closest_pair_indices)), closest_pair_indices][np.arange(len(pT_max_ind)), pT_max_ind].T
        self.Z2 = closest_pair.T[np.arange(len(pT_min_ind)),pT_min_ind]
        self.l3_calc, self.l4_calc = lepton_pairs.transpose(3,0,1,2)[np.arange(len(closest_pair_indices)), closest_pair_indices][np.arange(len(pT_min_ind)), pT_min_ind].T

        self.Z1_closest = closest_Z_pair.T[np.arange(len(closest_Z_min_ind)),closest_Z_min_ind]
        self.Z2_closest = closest_Z_pair.T[np.arange(len(closest_Z_max_ind)),closest_Z_max_ind]

        self.l1_calc_cls, self.l2_calc_cls = lepton_pairs.transpose(3,0,1,2)[np.arange(len(min_ind)), min_ind][np.arange(len(closest_Z_min_ind)), closest_Z_min_ind].T
        self.l3_calc_cls, self.l4_calc_cls = lepton_pairs.transpose(3,0,1,2)[np.arange(len(min_ind)), min_ind][np.arange(len(closest_Z_max_ind)), closest_Z_max_ind].T

        self.Z1_true = true_pair[0]
        self.Z2_true = true_pair[1]

        # Set Higgs four momentum
        self.H = self.Z1 + self.Z2


    def filter_Z(self, range1: tuple[int,int]=(50,115), range2: tuple[int,int]=(50,115)) -> None:
        """
        Filter all the arrays by allowed Z boson masses. Ranges for allowed masses are specified by range1 and range2.

        params:
            range1(Tuple[int,int]): The range for allowed Z1 masses. Default is 50 GeV <= Z1 <= 115 GeV 
            range2(Tuple[int,int]): The range for allowed Z2 masses. Default is 50 GeV <= Z2 <= 115 GeV 
        """
        cond1 = np.where((self.Z1.mass>=range1[0])&(self.Z1.mass<=range1[1]))
        cond2 = np.where((self.Z2.mass>=range2[0])&(self.Z2.mass<=range2[1]))

        # Get only indices where cond1 and cond2 apply
        indices = np.intersect1d(cond1,cond2)

        # Filter dataframe and sample
        self.df = self.df.take(indices)
        self.sample.events = self.df

        # Filter particle momenta
        self.Z1 = self.Z1[indices]
        self.Z2 = self.Z2[indices]
        self.H = self.H[indices]

        self.Z1_closest = self.Z1_closest[indices]
        self.Z2_closest = self.Z2_closest[indices]

        self.Z1_true = self.Z1_true[indices]
        self.Z2_true = self.Z2_true[indices]

        self.g1 = self.g1[indices]
        self.g2 = self.g2[indices]

        self.l1 = self.l1[indices]
        self.l2 = self.l2[indices]
        self.l3 = self.l3[indices]
        self.l4 = self.l4[indices]

        self.l1_calc = self.l1_calc[indices]
        self.l2_calc = self.l2_calc[indices]
        self.l3_calc = self.l3_calc[indices]
        self.l4_calc = self.l4_calc[indices]

        self.l1_calc_cls = self.l1_calc_cls[indices]
        
        self.l2_calc_cls = self.l2_calc_cls[indices]
        self.l3_calc_cls = self.l3_calc_cls[indices]
        self.l4_calc_cls = self.l4_calc_cls[indices]

        # Filter angle calc
        #self.Z1_angles = self.Z1_angles[indices]
        #self.Z2_angles = self.Z2_angles[indices]
        #self.Z1_true_angles = self.Z1_true_angles[indices]
        #self.Z2_true_angles = self.Z2_true_angles[indices]

        


    def get_kinematics(self, l1: MomentumObject4D=None, l2: MomentumObject4D=None, l3: MomentumObject4D=None, l4: MomentumObject4D=None, tensorize: bool=False) -> np.ndarray:
        """
        Calculates kinematic variables cos 𝜃∗, cos 𝜃1, cos 𝜃2, 𝜙1 ,𝜙, mZ1 and mZ2. Angles used are described in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.86.095031.
        If leptons are not supplied, the leptons from Z boson reconstruction will be used.

        params:
            l1 (MomentumObject4D): Lepton 1 used in calculations. Must be positive and paired up with l2 to form a Z boson. Can also be a vector array of MomentumObject4D objects.
            l2 (MomentumObject4D): Lepton 2 used in calculations. Must be negative and paired up with l1 to form a Z boson. Can also be a vector array of MomentumObject4D objects.
            l3 (MomentumObject4D): Lepton 3 used in calculations. Must be positive and paired up with l4 to form a Z boson. Can also be a vector array of MomentumObject4D objects.
            l4 (MomentumObject4D): Lepton 4 used in calculations. Must be negative and paired up with l3 to form a Z boson. Can also be a vector array of MomentumObject4D objects.
            tensorize (bool): If True, the output will be converted to a tensorflow tensor. False by default.

        Returns:
            np.ndarray: The kinematic variables in form of [cos 𝜃∗, cos 𝜃1, cos 𝜃2, 𝜙1 ,𝜙, mZ1, mZ2].
        """

        # Lazy, but use default behaviour when not all lepton momenta are supplied
        if l1 is None or l2 is None or l3 is None or l4 is None:
            l1 = self.l1_calc
            l2 = self.l2_calc
            l3 = self.l3_calc
            l4 = self.l4_calc
            Z1 = self.Z1
            Z2 = self.Z2
        else:
            Z1 = l1 + l2
            Z2 = l3 + l4

        # Boost both Z bosons to Higgs frame
        Z1_h = Z1.boost(-self.H)
        Z2_h = Z2.boost(-self.H)

        # Unit vector (3D) from boosted Z1 boson
        z1 = Z1_h.to_3D().unit()

        # Angle between Z1 and the beam axis
        cth_star = z1.z

        # Boost leptons to Higgs frame
        l1_h = l1.boost(-self.H)
        l2_h = l2.boost(-self.H)
        l3_h = l3.boost(-self.H)
        l4_h = l4.boost(-self.H)

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

        # TODO: decide what to do if n12 and n34 are parallel (now: removed from data)
        
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
            return tf.convert_to_tensor(np.array([cth_star, cth1, cth2, phi1, phi, self.Z1.mass, self.Z2.mass]).T)
        else:
            return np.array([cth_star, cth1, cth2, phi1, phi, self.Z1.mass, self.Z2.mass]).T


    def get_true_kinematics(self, tensorize: bool=False) -> np.ndarray:
        """
        Calculate the kinematics using Z boson pairs provided by the simulation data.

        Returns:
            np.ndarray: The kinematic variables in form of [cos 𝜃∗, cos 𝜃1, cos 𝜃2, 𝜙1 ,𝜙, mZ1, mZ2].
        """
        return self.get_kinematics(self.l2, self.l1, self.l4, self.l3, tensorize=tensorize)
        
    def get_closest_kinematics(self, tensorize: bool=False) -> np.ndarray:
        """
        Calculate the kinematics using Z boson pairs reconstructed by the closestZ approach.

        Returns:
            np.ndarray: The kinematic variables in form of [cos 𝜃∗, cos 𝜃1, cos 𝜃2, 𝜙1 ,𝜙, mZ1, mZ2].
        """
        return self.get_kinematics(self.l1_calc_cls, self.l2_calc_cls, self.l3_calc_cls, self.l4_calc_cls, tensorize=tensorize)