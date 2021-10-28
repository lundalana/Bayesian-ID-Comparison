"""
File Name: varFam.py
Purpose: This file contains the base variational family class used to 
            enact variational inference, as well as several children 
            classes which define specific parameterized distributions 
            for the variational family to follow. 
Developed by: Alana Lund (Purdue University) 
Last Updated: 26 Oct. 2021 
License: AGPL-3.0
"""

########## Import Statements ########## 
## Standard Imports ##
import abc
import numpy as np

## Torch Functions ##
import os
import torch
import torch.distributions.constraints as constraints
torch.set_default_dtype(torch.float64) # Set default type to float64 

################################################################################
########### Classes to Define Potential Variational Family Components ##########
################################################################################

class VariationalFamilies(abc.ABC):
    """
    Defines the variational family components to be used in the variational 
    inference problem.
    """

    @abc.abstractmethod
    def __init__(self,size,storage,*parameters):
        """
        Generates the initializes the gradient tracking on the variational 
        parameters. Also generates storage for the parameters over the 
        optimization interval. 
        size - number of variational parameters
        storage - number of terms to store in the optimization history. Storage
                  created will be +1 from number entered, as initial values is stored.
        parameters - initial values of the variational parameters
        """
        # Set Variational Parameters
        self.vp = torch.cat(parameters, dim=0)
        self.vp.requires_grad = True
        self.size = size

        # Create Storage for Optimization History 
        self.vpHist = np.zeros((storage+1, len(self.vp)))
        self.vpHist[0] = self.vp.detach().reshape(1,-1).numpy()[0,:]

    @abc.abstractmethod
    def rsample(self, n=1):
        """
        This function takes a random sample of the standard form of the 
        distribution and transforms it to the parameterized version of the
        distribution. Due to the use of gradients in this problem, this is 
        essentially an implementation of elliptical standardization. 
        Returns an nxSize tensor
        n - number of samples requested (scalar)
        """
        pass

    @abc.abstractmethod
    def entropy(self):
        """
        Calculates entropy of the variational family given the variational
        parameters in the current iteration.
        Returns a scalar
        """
        pass

    def store(self, ind):
        """
        Stores the current value of the variational parameters in the 
        history array. 
        ind - index of the array at which to store (scalar)
        """
        self.vpHist[ind] = self.vp.detach().reshape(1,-1).numpy()[0,:]

    @abc.abstractmethod
    def buildMat(self, ind):
        """
        Builds the L matrix at a particular index of the history.
        ind - index of the inference history (scalar)
        Returns size x size L-matrix [torch tensor]
        """
        pass

    @abc.abstractmethod
    def reset(self, storage, varPar):
        """
        storage - how much storage to create to track the inference 
                history (scalar)
        varPar - variational parameters to reset the values in the instance.
        """
        pass

    def __str__(self):
        return "Variational Family Base Class"

class NormalDiag(VariationalFamilies):
    """
    Defines the Gaussian variational component with independent terms to be used
    in the variational inference problem. 
    """
    def __init__(self, size, storage, m=None, log_s=None):
        """
        m - starting mean (1 x size)
        log_s - starting log standard deviation vector (1 x size)
        Further details in the parent class. 
        """
        # Check User Input
        if m is None:
            m = torch.randn(size) 
        else:
            if len(m) != size:
                raise IndexError('m must be of shape [1 x size]')
            if not torch.is_tensor(m):
                raise TypeError('m must be a torch tensor')

        if log_s is None:
            log_s = torch.randn(size) 
        else:
            if len(log_s) != size:
                raise IndexError('s must be of shape [1 x size]')
            if not torch.is_tensor(log_s):
                raise TypeError('s must be a torch tensor')

        # Record Variational Parameters and Set Storage
        return super().__init__(size,storage,m,log_s)

    def rsample(self, n=1):
        """
        See parent class for function details. 
        """
        return (self.vp[:self.size] + torch.randn(n,self.size) 
                        * self.vp[self.size:].exp())

    def entropy(self):
        """
        See parent class for function details.
        """
        return (torch.tensor(self.size/2*np.log(2*np.pi) + self.size/2) 
            + self.vp[self.size:].sum())

    def buildMat(self, ind=None):
        """
        See parent class for function details.
        """
        if ind == None:
            a = self.vp[self.size:].detach()
        else:
            a = torch.tensor(self.vpHist[ind,self.size:])

        return torch.diag(a.exp())

    def reset(self, storage, varPar=None):
        """
        varPar - [1 x VPLen] consists of 
            m - starting mean (1 x size)
            log_s - starting log standard deviation vector (1 x size)
            Further details in the parent class. 
        """
        if varPar == None: 
            # Record Variational Parameters and Set Storage
            m = self.vp[:self.size].detach()
            log_s = self.vp[self.size:].detach()
            return super().__init__(self.size,storage,m,log_s)
       
        else:
            # Check User Input  
            if len(varPar) != (2*self.size):
                raise IndexError('Input must be of shape [1 x 2*size]')
            if not torch.is_tensor(varPar):
                raise TypeError('Input must be a torch tensor')

            m = varPar[:self.size]
            log_s = varPar[self.size:]

            # Record Variational Parameters and Set Storage
            return super().__init__(self.size,storage,m,log_s)

    def __str__(self):
        return "Gaussian variational family assuming all terms are independent"

class NormalTri(VariationalFamilies):
    """
    Defines the Gaussian variational component assuming that adjacent terms 
    are correlated. 
    """
    def __init__(self, size, storage, m=None, s=None):
        """
        m - starting mean (1 x size)
        s - starting L-matrix, cast as a vector (1 x [2*size - 1])
            Note: this vector is ordered such that the main diagonal of L is 
            recorded first and the lower diagonal of L is appended. 
        Further details in the parent class.
        """
        # Check User Input
        if m is None:
            m = torch.randn(size) 
        else:
            if len(m) != size:
                raise IndexError('m must be of shape [1 x size]')
            if not torch.is_tensor(m):
                raise TypeError('m must be a torch tensor')

        if s is None:
            s = torch.randn(int(2*size-1)) 
        else:
            if len(s) != (2*size-1):
                raise IndexError('s must be of shape [1 x (2*size-1)]')
            if not torch.is_tensor(s):
                raise TypeError('s must be a torch tensor')

        # Record Variational Parameters and Set Storage
        return super().__init__(size,storage,m,s)

    def rsample(self, n=1):      
        """
        See parent class for function details. 
        """
        return self._tri_mult(torch.randn((self.size,n)))+self.vp[:self.size]

    def entropy(self):
        """
        Calculates entropy of the variational family given the variational
        parameters in the current iteration.
        Returns a scalar
        """
        return (torch.tensor(self.size/2*np.log(2*np.pi) + self.size/2) 
            + 0.5* torch.log(torch.pow(self.vp[self.size:2*self.size], 2)).sum())

    def buildMat(self, ind=None):
        """
        See parent class for function details.
        """
        if ind == None:
            a = self.vp[self.size:].detach()
        else:
            a = torch.tensor(self.vpHist[ind,self.size:])

        return a[:self.size].diag() + a[self.size:].diag(-1)

    def reset(self, storage, varPar=None):
        """
        varPar - [1 x VPLen] consists of 
            m - starting mean (1 x size)
            s - starting L-matrix, cast as a vector (1 x [2*size - 1])
                Note: this vector is ordered such that the main diagonal of L is 
                recorded first and the lower diagonal of L is appended. 
            Further details in parent class. 
        """
        if varPar == None: 
            # Record Variational Parameters and Set Storage
            m = self.vp[:self.size].detach()
            s = self.vp[self.size:].detach()
            return super().__init__(self.size,storage,m,s)
       
        else:
            # Check User Input  
            if len(varPar) != (3*self.size-1):
                raise IndexError('Input must be of shape [1 x (3*self.size-1)]')
            if not torch.is_tensor(varPar):
                raise TypeError('Input must be a torch tensor')

            m = varPar[:self.size]
            s = varPar[self.size:]

            # Record Variational Parameters and Set Storage
            return super().__init__(self.size,storage,m,s)

    ################################ Tools #####################################

    def _tri_mult(self,samples):
        """
        Left-multiplies a bi-diagonal L matrix with a matrix of random samples
        samples - Nxsize vector where each column is a sample 
        Returns an Nxsize matrix, where each column is a sample 
        """
        
        # Get Vector Representing L matrix
        a = self.vp[self.size:]

        # Extract Dimensions
        states = self.size
        n = samples.size()[1]        # number of random sample vectors
        
        # Isolate Various Elements    
        diagVec = a[:states].reshape([-1,1])   # diagonal
        ldiagVec = torch.cat((torch.zeros(1), a[states:])).reshape([-1,1])  # lower diagonal
        bdiag = torch.cat((torch.zeros((1,n)),samples[:states-1,:]), dim = 0)   
                               # Random Matrix for multiplying lower diag
        
        output = diagVec*samples + ldiagVec*bdiag
            
        return torch.t(output)

    def __str__(self):
        return "Gaussian variational family assuming adjacent terms are dependent"

class Deterministic(VariationalFamilies):
    """
    Defines a constant value in the state-space model. This class is created to 
    homogenize the data types and allow for equivalent treatment in the 
    stochastic time-series model. 
    """
    def __init__(self, size, storage, m=None):
        """
        m - parameter values (1 x size)
        Further details in parent class
        """
        # Check User Input
        if m is None:
            raise ValueError("Deterministic class must receive input constants.")
        if len(m) != size:
            raise IndexError('m must be of shape [1 x size]')
        if not torch.is_tensor(m):
            raise TypeError('m must be a torch tensor')
        
        # Record Parameters
        self.vp = m   
        self.vp.requires_grad = False
        self.size = size

        # Create Storage for Optimization History 
        self.vpHist = np.zeros((storage+1, self.size))
        self.vpHist[0] = self.vp.detach().reshape(1,-1).numpy()[0,:]

    def rsample(self, n=1):
        """
        This function generates a nxsize tensor of the constants, for use in 
        the stochastic model of the time-series.
        n - number of samples requested (scalar)
        """
        return  self.vp.repeat(n,1)

    def entropy(self):
        """
        Deterministic variables have 0 entropy.
        """
        return 0

    def buildMat(self, ind=None):
        """
        Builds the L matrix at a particular index of the history.
        ind - index of the inference history (scalar)
            Note: if no index is provided, the current values are taken
        Returns size x size zero matrix, as constants have no variance
        """
        return torch.zeros((size,size))

    def reset(self, storage, varPar=None):
        """
        varPar - parameter values (1 x size)
        Further details in parent class
        """
        if varPar == None:
            pass
        else:
            # Check User Input
            if len(varPar) != self.size:
                raise IndexError('Input must be of shape [1 x size]')
            if not torch.is_tensor(varPar):
                raise TypeError('Input must be a torch tensor')
        
            # Record Parameters
            self.vp = varPar   
            self.vp.requires_grad = False

        # Create Storage for Optimization History 
        self.vpHist = np.zeros((storage+1, self.size))
        self.vpHist[0] = self.vp.detach().reshape(1,-1).numpy()[0,:]

    def __str__(self):
        return "Variational family to give homogeneous description to "  +     \
            "constants in the stochastic model."