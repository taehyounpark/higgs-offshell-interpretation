import vector
import numpy as np

from hstar.process import Sample

class ZmassPairChooser:
    def __init__(self, sample: Sample):
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

    def find_Z(self) -> np.ndarray:
        """
        Find the lepton pairs whose invariant masses are closest to the true Z boson rest mass by using chi squared minimization.

        Returns:
            Numpy array of lepton four-momenta where l1 and l2 form the Z1 boson and l3 and l4 form the Z2 boson. 
            Leptons l1 and l3 are negative while l2 and l4 are positive.
        """
        # Possible Z bosons from leptons 
        p12 = (self.l1 + self.l2)
        p14 = (self.l1 + self.l4)
        p23 = (self.l2 + self.l3)
        p34 = (self.l3 + self.l4)

        # Possible Z boson pairs as Momentum4D objects in vector arrays
        pairs = vector.array([[p12, p34], [p14, p23]], dtype=[('px',float),('py',float),('pz',float),('E',float)])
        lepton_pairs = vector.array([[[self.l1,self.l2],[self.l3,self.l4]],
                                      [[self.l1,self.l4],[self.l3,self.l2]]], dtype=[('px',float),('py',float),('pz',float),('E',float)])

        del p12
        del p14
        del p23
        del p34

        #print('Deleting refs p12, p23, p14, p34')

        # Chi squared minimization to determine the closest pair
        chi_sq = np.array([(pair[0].mass - self.Z_mass)**2 + (pair[1].mass - self.Z_mass)**2 for pair in pairs]).T
        closest_pair_indices = np.argmin(chi_sq, axis=1)
        closest_pair = pairs.transpose(2,0,1)[np.arange(len(closest_pair_indices)), closest_pair_indices].T

        del chi_sq

        #print('Deleting ref chi_sq')

        # Determine the Z boson with the higher pT
        # That one will be Z1, the other one Z2
        pT_max_ind = np.argmax(closest_pair.pt,axis=0) # Z1
        pT_min_ind = np.argmin(closest_pair.pt,axis=0) # Z2

        # Determine the order manually if both Z bosons have the same pT
        cond=(pT_max_ind==pT_min_ind)

        pT_max_ind[cond] = 0
        pT_min_ind[cond] = 1

        del cond

        #print('Deleting ref cond')

        # Set Z1, Z2 and (l1_calc, l2_calc) = Z1; (l3_calc, l4_calc) = Z2
        self.Z1 = closest_pair.T[np.arange(len(pT_max_ind)),pT_max_ind]
        self.l1_calc, self.l2_calc = lepton_pairs.transpose(3,0,1,2)[np.arange(len(closest_pair_indices)), closest_pair_indices][np.arange(len(pT_max_ind)), pT_max_ind].T
        self.Z2 = closest_pair.T[np.arange(len(pT_min_ind)),pT_min_ind]
        self.l3_calc, self.l4_calc = lepton_pairs.transpose(3,0,1,2)[np.arange(len(closest_pair_indices)), closest_pair_indices][np.arange(len(pT_min_ind)), pT_min_ind].T

        del pairs
        del lepton_pairs
        del closest_pair_indices
        del closest_pair
        del pT_max_ind
        del pT_min_ind

        #print('Deleting refs pairs, lepton_pairs, closest_pair_indices, closest_pair, pT_max_ind, pT_min_ind')

        # Set Higgs four momentum
        self.H = self.Z1 + self.Z2

        self.filter_Z()

        #print(np.sum((self.Z1 == self.Z2).astype(int)))

        return vector.array([self.l1_calc,self.l2_calc,self.l3_calc,self.l4_calc], dtype=[('px',float),('py',float),('pz',float),('E',float)]).T

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

        self.m4l = self.m4l[indices]


class ClosestZmassChooser:
    def __init__(self, sample: Sample):
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


    def find_Z(self) -> np.ndarray:
        """
        Find the lepton pairs whose invariant masses are closest to the true Z boson rest mass by determining the closest pair to be Z1 and select the opposite pair to be Z2.

        Returns:
            Numpy array of lepton four-momenta where l1 and l2 form the Z1 boson and l3 and l4 form the Z2 boson. 
            Leptons l1 and l3 are negative while l2 and l4 are positive.
        """
        # Possible Z bosons from leptons 
        p12 = (self.l1 + self.l2)
        p14 = (self.l1 + self.l4)
        p23 = (self.l2 + self.l3)
        p34 = (self.l3 + self.l4)

        # Possible Z boson pairs as Momentum4D objects in vector arrays
        pairs = vector.array([[p12, p34], [p14, p23]], dtype=[('px',float),('py',float),('pz',float),('E',float)])
        lepton_pairs = vector.array([[[self.l1,self.l2],[self.l3,self.l4]],
                                      [[self.l1,self.l4],[self.l3,self.l2]]], dtype=[('px',float),('py',float),('pz',float),('E',float)])

        # Just choose the Z boson pair which contains the Z boson closest to the true rest mass
        pairs_diffs = ((pairs.mass - np.ones(pairs.shape)*self.Z_mass)**2).transpose(2,0,1).reshape(pairs.shape[2],4)
        min_ind = np.floor(np.argmin(pairs_diffs, axis=1)/2.0).astype(int)
        closest_Z_pair = pairs.transpose(2,0,1)[np.arange(len(min_ind)),min_ind].T

        closest_Z_min_ind = np.argmin((closest_Z_pair.mass-self.Z_mass)**2, axis=0)
        closest_Z_max_ind = np.argmax((closest_Z_pair.mass-self.Z_mass)**2, axis=0)

        # Set Z1, Z2 and (l1_calc, l2_calc) = Z1; (l3_calc, l4_calc) = Z2
        self.Z1 = closest_Z_pair.T[np.arange(len(closest_Z_min_ind)),closest_Z_min_ind]
        self.l1_calc, self.l2_calc = lepton_pairs.transpose(3,0,1,2)[np.arange(len(min_ind)), min_ind][np.arange(len(closest_Z_min_ind)), closest_Z_min_ind].T
        self.Z2 = closest_Z_pair.T[np.arange(len(closest_Z_max_ind)),closest_Z_max_ind]
        self.l3_calc, self.l4_calc = lepton_pairs.transpose(3,0,1,2)[np.arange(len(min_ind)), min_ind][np.arange(len(closest_Z_max_ind)), closest_Z_max_ind].T

        # Set Higgs four momentum
        self.H = self.Z1 + self.Z2

        self.filter_Z()

        return vector.array([self.l1_calc,self.l2_calc,self.l3_calc,self.l4_calc], dtype=[('px',float),('py',float),('pz',float),('E',float)]).T
    
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

        self.m4l = self.m4l[indices]