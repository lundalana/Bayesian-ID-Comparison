"""
File Name: stochMod.py
Purpose: This file contains the Priors class, which develops functionalities 
            on the prior distributions necessary for variational inference. 
            The stochastic time series model class is also defined, which
            developes the log-likelihood and log-transition probability for 
            the stochastic model, as well as calculating the ELBO. Finally, 
            the "inferTimeSeries" is defined, which runs the stochastic 
            optimization and stores intermediate results over variable epochs
            of optimization. 
Developed by: Alana Lund (Purdue University) 
Last Updated: 26 Oct. 2021 
License: AGPL-3.0
"""

########## Import Statements ########## 
## Standard Imports ##
from __future__ import print_function
import abc
import numpy as np
import scipy as sp
import datetime

## Torch Functions ##
import os
import torch
import torch.distributions.constraints as constraints
torch.set_default_dtype(torch.float64) # Set default type to float64 

################################################################################
#######    Prior classes (Prior Distributions with Transformations)      #######
################################################################################

class Prior(abc.ABC):
    """
    This is the base class to define the priors on the inferred states for the
    variational inference problem. Each prior needs the ability to be 
    transformed to the Euclidean space and to take its logprior given a sample. 
    """

    @abc.abstractmethod
    def __init__(self, comp, dim, *params, name="Component", debug=False):
        """
        comp - name of the corresponding variational family component (str)
            Note: must match the name of the dictionary key for the guide 
            component
        dim - index of the term in the corresponding variational family 
            component (int)
        parameters - list of parameters required to define the prior distribution
                   - details in the child classes.
        name - the name of the variable that this prior models
        """
        self.comp = comp
        self.dim = dim
        self.par = params
        self.name = name
        self.debug = debug

    @abc.abstractmethod
    def logPrior(self, samples, varMean=None):
        """
        The average of the log-probabilities of the selected distribution 
        evaluated over a set of samples with full support. Samples are 
        transformed to the support of the prior during function operation. 

        Parameters
            samples - A torch tensor of samples on the prior (Nx1)
            varMean - mean of the variational family for the current iteration

        Returns - torch scalar
        """
        pass

    @abc.abstractmethod
    def transform(self, samples):
        """
        Transforms the set of samples from the support of the prior to support 
        in the Euclidean space. 

        Parameters
            samples - A 1-D torch tensor of samples on the prior 

        Returns - torch tensor (Nx1)
        """
        pass

    @abc.abstractmethod
    def invTransform(self, samples):
        """
        Transforms the set of samples from having support on the Euclidean space
        to having the same support as the prior.

        Parameters
            samples - A 1-D torch tensor of samples on the prior 

        Returns - torch tensor (Nx1)
        """
        pass

    @abc.abstractmethod
    def sample(self, size):
        """
        Produces a set of samples of the random variable of shape 'size'

        Parameters
            size - tuple defining the dimension of the output tensor of random
                   samples to produce (note, if dimension is 1, specify (N,))

        Returns - tensor of random samples of shape 'size'
        """
        pass

    @abc.abstractmethod
    def plot(self, x, *posterior, modes=False):
        """
        Evaluates the prior and posterior probabilities of the random variable 
        over the set of points x. Posterior probabilities are calculated 
        according to the transformations expressed in the child class, as it is 
        assumed that the posterior is generated through ADVI. 

        Parameters
        x - 1-D numpy array of points to evaluate the distribution over 
        posterior - list of parameters required to define the prior distribution
            Note: Parameters must be listed in increasing order
               (e.g. central moment-->Xth central moment, lower --> upper bound)

        Returns - tuple of numpy arrays with the probability values of the 
                  samples over the prior and posterior distributions 
                  (yprior, yposterior)
        """
        pass

    def __str__(self):
        return "base class of prior for %s" %self.name

class Normal(Prior):
    """
    This class defines normally distributed priors on the inferred states 
    for the variational inference problem.
    """

    def __init__(self, comp, dim, *params, name="Component", debug=False):
        """
        par[0] - mean
        par[1] - standard deviation
        Further details in the parent class
        """
        self.mean = params[0]
        self.var = params[1]**2
        self.transMean = self.mean
        self.transVar = self.var 

        # Record Variational Parameters and Set Storage
        return super().__init__(comp, dim, *params, name=name, debug=debug)

    def logPrior(self, samples, varMean=None):
        """
        The average of the log-probabilities of the Normal distribution 
        evaluated over the given samples. Further details in parent class. 
            Note: varMean not required in this instance
        """
        nsamp = len(samples)
        logPrior = torch.distributions.Normal(self.par[0], 
            self.par[1]).log_prob(self.invTransform(samples))
        
        if self.debug:
            print("Log of the Normal Prior Distribution = \n" +                \
                str(logPrior) + "\n") 
        
        return torch.sum(logPrior)/nsamp

    def transform(self, samples):
        """
        Normal priors have full support on the Euclidean space. Further details 
        in parent class.
        """
        return samples.reshape(-1,1)

    def invTransform(self, samples):
        """
        Normal priors have full support on the Euclidean space. Further details 
        in parent class.
        """
        return samples.reshape(-1,1)

    def sample(self, size):
        """
        Further details in parent class.
        """
        return torch.distributions.Normal(self.par[0], self.par[1]).sample(size)

    def plot(self, x, *posterior, modes=False):
        """
        x - points to evaluate the distribution over (N)
        posterior - [0] mean of posterior, [1] standard deviation of posterior
        name - Name of the variable we're studying
        """
        yPrior=sp.stats.norm.pdf(x, loc=self.par[0], scale=self.par[1])
        yPost=sp.stats.norm.pdf(x, loc=posterior[0], scale=posterior[1])
        
        if modes == True:
            print("Posterior Mean of %s = %f" %(self.name,posterior[0]) )
            print("Posterior Standard Deviation of %s = %f"%(self.name,posterior[1]))
        
        return (yPrior,yPost)

    def __str__(self):
        return "Normal distribution of %s with mean %.4f and standard "        \
        "deviation %.4f" %(self.name,*self.par)

class LogNormal(Prior):
    """
    This class defines lognormally distributed priors on the inferred states 
    for the variational inference problem.
    """

    def __init__(self, comp, dim, *params, name="Component", debug=False):
        """
        par[0] - mean
        par[1] - standard deviation
        Further details in the parent class
        """
        self.mean = np.exp(params[0] + params[1]**2/2)
        self.var = (np.exp(params[1]**2)-1)*np.exp(2*params[0] + params[1]**2)
        self.transMean = params[0]
        self.transVar = params[1]**2

        # Record Variational Parameters and Set Storage
        return super().__init__(comp, dim, *params, name=name, debug=debug)

    def logPrior(self, samples, varMean=None):
        """
        The average of the log-probabilities of the LogNormal distribution 
        evaluated over the given samples. Further details in parent class. 
        """
        nsamp = len(samples)
        logPrior = torch.distributions.LogNormal(self.par[0], 
            self.par[1]).log_prob(self.invTransform(samples))
        
        if self.debug:
            print("Log of the LogNormal Prior Distribution = \n" +             \
                str(logPrior) + "\n") 
        
        return torch.sum(logPrior)/nsamp + varMean

    def transform(self, samples):
        """
        Transforms LogNormal samples, which have support on the positive reals, 
        to have support on the Euclidean space. Further details in parent class.
        """
        return samples.reshape(-1,1).log()

    def invTransform(self, samples):
        """
        Transforms samples from the Euclidean space to the support of the 
        LogNormal distribution (positive reals). Further details in parent class.
        """
        return samples.reshape(-1,1).exp()

    def sample(self, size):
        """
        Further details in parent class.
        """
        return torch.distributions.LogNormal(self.par[0],                  \
            self.par[1]).sample(size)

    def plot(self, x, *posterior, modes=False):
        """
        posterior - [0] mean of posterior, [1] standard deviation of posterior
        Further details in parent class.
        """
        yPrior=sp.stats.lognorm.pdf(x, self.par[1], scale=np.exp(self.par[0]))
        yPost=sp.stats.lognorm.pdf(x, posterior[1],                            \
            scale=np.exp(posterior[0]))
        
        muPost = np.exp(posterior[0] + np.square(posterior[1])/2)
        stdPost = np.sqrt((np.exp(np.square(posterior[1]))-1)* 
            np.exp(2*posterior[0] + np.square(posterior[1])))

        if modes == True:
            print("Posterior Mean of %s = %f" %(self.name,muPost) )
            print("Posterior Standard Deviation of %s = %f"%(self.name,stdPost))
        
        return (yPrior,yPost)

    def __str__(self):
        return "LogNormal distribution of %s with mean %.4f and standard"      \
        " deviation %.4f" %(self.name,*self.par)

class Deterministic(Prior):
    """
    This class defines deterministic parameters for the variational inference 
    problem in a manner which is homogeneous with the inferred states.
    par[0] - value
    """
    def __init__(self, comp, dim, *params, name="Component", debug=False):
        """
        par[0] - rate
        Further details in the parent class
        """
        self.mean = params[0]

        # Record Variational Parameters and Set Storage
        return super().__init__(comp, dim, *params, name=name, debug=debug)

    def logPrior(self, samples, varMean=None):
        """
        As there is absolute certainty in this value, it's probability is 1. 
        """        
        return 0

    def transform(self, samples):
        """
        Deterministic priors require no transformation.
        """
        return samples.reshape(-1,1)

    def invTransform(self, samples):
        """
        Deterministic priors require no transformation.
        """
        return samples.reshape(-1,1)

    def sample(self, size):
        """
        Returns an torch tensor filled with the constant value. 
        """
        return torch.ones(size)*self.par[0]

    def plot(self, x, *posterior):
        """
        A deterministic prior denotes a constant value in the problem. The
        probability is therefore infinite at the value and 0 everywhere else. 
        For simplicity (and plot-ability), we show infinity as 1. 
        """
        yPrior= (x == self.par[0]).astype(float)
        yPost = yPrior

        print("The constant parameter %s = %f" %(self.name,self.par[0]) )
        
        return (yPrior,yPost)

    def __str__(self):
        return "Deterministic parameter %s with value %.4f "                   \
            %(self.name,*self.par)

################################################################################
##### Stochastic Models of the Types of Inference we're trying to Achieve ######
################################################################################

class StochasticTimeSeriesModels(object):
    """
    This is the base class to represent a stochastic model of a time-series 
    problem, meaning that we infer the states and parameters of a dynamical 
    system. 
    """

    def __init__(self,dt,priors,debug=False):
        """
        dt - sampling frequency of data collection (dimensional) (scalar)
        priors - dictionary of moments of the inferred states and parameters. 
            Should include all parameters of the model, even those which are 
            deterministic. Excludes the states described by the likelihood and 
            transition probabilities which this class defines. 
            e.g. state - central moments of the 
            prior distribution on the states (Sx2 tensor)
        """
        self.dt = torch.tensor(dt, dtype = torch.double)
        for key in priors:
            if not isinstance(priors[key],Prior):
                print(priors[key],"is not of type prior")
                #raise TypeError("All distributions must be of type 'prior'")
        self.priors = priors
        self.debug = debug

    def logLike(self, data, **samples):
        """
        Takes the log-likelihood of the data over the given sample set. 
        data - system responses (nMeas x N)
        samples - dictionary of samples over the inferred variables
        """
        raise ModuleNotFoundError("base class should not be called!")

    def logTrans(self, base, **samples):
        """
        Takes the log-transition probability of the system states over the 
        given sample set. 
        base - system inputs (nInp x N)
        samples - dictionary of samples over the inferred variables
        """
        raise ModuleNotFoundError("base class should not be called!")

    ################################ Tools #####################################

    def elbo(self, guide, data, base, nsamp, annealFactor):
        """
        The evidence lower bound of the variational inference problem, defined
        over the stochastic model of the system and the variational family.         
        guide - a dictionary defining the components of the variational family
        data - the observations of the system (N)
        base - input force for the system (N)
        nsamp - the number of MC samples to take (scalar)
        """

        ## Generate Samples ##
        # Generates samples on each component of the guide and places them in
        # a dictionary (samples) with the same key as guide
        samples = {}
        for key in guide:
            samples[key] = guide[key].rsample(nsamp)

        ## Calculate ELBO ##
        # Calculates the stochastic model specific terms by evaluating the log-
        # likelihood and the log-transition probability over the generated samples
        logLikeFactor = self.logLike(data, **samples)
        logTransFactor = self.logTrans(base, **samples)

        # Calculates the log-priors using the specified prior distributions over
        # the generated samples
        priorsFactor = 0
        for key in self.priors:
            p = self.priors[key]
            varMean = guide[p.comp].vp[p.dim]
            logPriorAdd = p.logPrior(samples[p.comp][:,p.dim], varMean)
            priorsFactor += logPriorAdd

            if self.debug:
                print("LogPrior = \n" + str(logPriorAdd) + "\n")

        # Calculates the entropy over each variational component
        entFactor = 0
        for key in guide:
            ent = guide[key].entropy()
            entFactor += ent

            if self.debug:
                print("Variational Parameters = \n" + str(guide[key].vp))
                print("Entropy = \n" + str(ent) + "\n")

        return logLikeFactor + annealFactor*(logTransFactor+priorsFactor+entFactor)
    
    def __str__(self):
        return "Base class for stochastic time series models."

class LSDOF_Euler(StochasticTimeSeriesModels):
    """
    This defines a stochastic time-series model for variational inference and 
    takes its elbo. In this specific case, our system is a single-degree-of-
    freedom oscillator in which all states, parameters, and process noise terms
    are inferred. The potential inferred parameters are listed below. If a 
    parameter is not inferred, it must be given a deterministic variational
    family component. 

    priors['x1'] - prior distirbution on initial displacement
                 - related to variational component guide['disp'], component 0
    priors['x2'] - prior distribution on initial velocity
                 - related to variational component guide['vel'], component 0
    priors['xi'] - prior on the damping ratio parameter
                 - related to variational component guide['par'], component 0
    priors['wn'] - prior on the natural frequency parameter
                 - related to variational component guide['par'], component 1
    priors['w1'] - prior on the process noise parameter on disp
                 - related to variational component guide['Qnoise'], component 0
    priors['w2'] - prior on the process noise parameter on vel
                 - related to variational component guide['Qnoise'], component 1
    priors['v']  - prior on the measurement noise parameter 
                 - related to variational component guide['Rnoise'], component 0
    """

    def logLike(self, data, **samples):
        """
        Takes the log-likelihood of the data over the given sample set. 
        data - system responses (nMeas x N)
        samples['disp'] - samples on the disp. variational component (MCsamp x N)
        samples['vel'] - samples on the vel. variational component (MCsamp x N)
        samples['par'] - samples on the parameter variational component 
                        (MCsamp x Npar)
        samples['Qnoise'] - samples on the noise variational component 
                        (MCsamp x Nnoise_Q)
        samples['Rnoise'] - samples on the noise variational component 
                        (MCsamp x Nnoise_R)
        """
        dispKey = self.priors['x1'].comp
        velKey = self.priors['x2'].comp
        (nSamp, nState) = samples[dispKey].shape

        tranSamp = {}
        for key in self.priors:
            p = self.priors[key]
            tranSamp[key] = p.invTransform(samples[p.comp][:,p.dim])

        states = torch.stack((samples[dispKey], samples[velKey]), dim=0)
        par = torch.stack((tranSamp['xi'], tranSamp['wn']), dim=0)

        obsMean = self.stochSDOF(states, par)[1]

        logL = torch.distributions.Normal(obsMean, tranSamp['v']).log_prob(data)
        
        if self.debug:
            print("Parameters = \n" + str(par) + "\n")
            print("Mean of the Observations = \n" + str(obsMean) + "\n")
            print("Log Likelihood = \n" + str(logL) + "\n")

        return torch.sum(logL)/nSamp

    def logTrans(self, base, **samples):
        """
        Takes the log-transition probability of the states over the given input. 
        base - input excitation (nInp x N)
        samples['disp'] - samples on the disp. variational component (MCsamp x N)
        samples['vel'] - samples on the vel. variational component (MCsamp x N)
        samples['par'] - samples on the parameter variational component 
                        (MCsamp x Npar)
        samples['Qnoise'] - samples on the noise variational component 
                        (MCsamp x Nnoise_Q)
        samples['Rnoise'] - samples on the noise variational component 
                        (MCsamp x Nnoise_R)
        """

        dispKey = self.priors['x1'].comp
        velKey = self.priors['x2'].comp

        nData = len(base)
        (nSamp, nState) = samples[dispKey].shape
        bSize = int(nData/nState)

        tranSamp = {}
        for key in self.priors:
            p = self.priors[key]
            tranSamp[key] = p.invTransform(samples[p.comp][:,p.dim])

        states = torch.stack((samples[dispKey][:,:-1], 
                                samples[velKey][:,:-1]), dim=0)
        par = torch.stack((tranSamp['xi'], tranSamp['wn']), dim=0)
        exc = base[:nData-bSize]

        for i in range(bSize):
            states = self.stochEuler(self.stochSDOF, states, par, exc[i::bSize]) 

        ## Transition Probability on X1 ##
        logX1= torch.distributions.Normal(states[0],                           \
                    tranSamp['w1']).log_prob(samples[dispKey][:,1:])
        
        ## Transition Probability on X2 ##
        logX2 = torch.distributions.Normal(states[1],                          \
                    tranSamp['w2']).log_prob(samples[velKey][:,1:])

        
        if self.debug:
            print("Number of Data Points = \n" + str(nData) + "\n")
            print("Number of Samples = \n" + str(nSamp) + "\n")
            print("Number of States = \n" + str(nState) + "\n")
            print("Batch Size = \n" + str(bSize) + "\n")
            print("The shape of the input Matrix is \n" + str(base.shape) + "\n")
            print("The final displacement is \n" + str(states[0]) +"\n")
            print("The final velocity is \n" + str(states[1]) +"\n")
            print("Log of the Transition Probability on X1 = \n" + str(logX1) + "\n")
            print("Log of the Transition Probability on X2 = \n" + str(logX2) + "\n")
        
        return (torch.sum(logX1) + torch.sum(logX2))/nSamp

    # def elbo(self, guide, data, base, nsamp): # Fully Defined by Parent Class

    ################################# Tools #################################
    def stochEuler(self, model, states, par, inp):
        return states + self.dt*model(states,par, exc=inp)

    def stochSDOF(self, states, par, exc=None):
        if exc is None:
            exc = torch.zeros(states[0].shape)

        output = torch.stack((states[1], -2*par[0]*par[1]*states[1]
                           -(par[1]**2)*states[0]-exc), dim=0)
        return output

    def __str__(self):
        return "Linear single-degree-of-freedom stochastic time series model"

class BWSDOF_Euler(StochasticTimeSeriesModels):
    """
    This defines a stochastic time-series model for variational inference and 
    takes its elbo. In this specific case, our system is a single-degree-of-
    freedom oscillator in which all states, parameters, and process noise terms
    are inferred. The potential inferred parameters are listed below. If a 
    parameter is not inferred, it must be given a deterministic variational
    family component. 

    priors['x1'] - prior distirbution on initial displacement
                 - related to variational component guide['disp'], component 0
    priors['x2'] - prior distribution on initial velocity
                 - related to variational component guide['vel'], component 0
    priors['x3'] - prior distribution on initial Bouc-Wen displacement
                 - related to variational component guide['rdisp'], component 0
    priors['xi'] - prior on the viscous damping parameter
                 - related to variational component guide['par'], component 0
    priors['wn'] - prior on the friction damping parameter
                 - related to variational component guide['par'], component 1
    priors['beta'] - prior on the linear stiffness parameter
                 - related to variational component guide['par'], component 2
    priors['n'] - prior on the nonlinear stiffness parameter
                 - related to variational component guide['par'], component 3
    priors['gamma'] - prior on the nonlinear stiffness parameter
                 - related to variational component guide['par'], component 4
    priors['v']  - prior on the measurement noise parameter 
                 - related to variational component guide['constants'], component 0
    priors['w1'] - prior on the process noise parameter on disp
                 - related to variational component guide['constants'], component 1
    priors['w2'] - prior on the process noise parameter on vel
                 - related to variational component guide['constants'], component 2
    priors['w3'] - prior on the process noise parameter on vel
                 - related to variational component guide['constants'], component 3
    priors['xa'] - nondimensional length scale
                 - related to variational component guide['constants'], component 4
    priors['wa'] - nondimensional time scale
                 - related to variational component guide['constants'], component 5
    """
    def logLike(self, data, **samples):
        """
        Takes the log-likelihood of the data over the given sample set. 
        data - system responses (nMeas x N)
        samples['disp'] - samples on the disp. variational component (MCsamp x N)
        samples['vel'] - samples on the vel. variational component (MCsamp x N)
        samples['par'] - samples on the parameter variational component 
                        (MCsamp x Npar)
        samples['Qnoise'] - samples on the noise variational component 
                        (MCsamp x Nnoise_Q)
        samples['Rnoise'] - samples on the noise variational component 
                        (MCsamp x Nnoise_R)
        """
        dispKey = self.priors['x1'].comp
        velKey = self.priors['x2'].comp
        rdispKey =  self.priors['x3'].comp
        (nSamp, nState) = samples[dispKey].shape

        tranSamp = {}
        for key in self.priors:
            p = self.priors[key]
            tranSamp[key] = p.invTransform(samples[p.comp][:,p.dim])

        ## Calculate Log Likelihood on Acc ##
        states = torch.stack((samples[dispKey], samples[velKey], 
            samples[rdispKey]), dim=0)
        par = torch.stack((tranSamp['xi'], tranSamp['wn'],
                            tranSamp['beta'], tranSamp['n'], 
                            tranSamp['gamma']), dim=0)
        
        accMean = self.stochBW(states, par)[1]
        logAcc =torch.distributions.Normal(accMean,
                                        tranSamp['v']).log_prob(data)
      
        if self.debug:
            print("Mean of the Acc = \n" + str(accMean.shape) + "\n")
            print("Log Likelihood of the Acc = \n" + str(logAcc.shape) + "\n")
            
        return torch.sum(logAcc)/nSamp

    def logTrans(self, base, **samples):
        """
        Takes the log-transition probability of the states over the given input. 
        base - input excitation (nInp x N)
        samples['disp'] - samples on the disp. variational component (MCsamp x N)
        samples['vel'] - samples on the vel. variational component (MCsamp x N)
        samples['par'] - samples on the parameter variational component 
                        (MCsamp x Npar)
        samples['Qnoise'] - samples on the noise variational component 
                        (MCsamp x Nnoise_Q)
        samples['Rnoise'] - samples on the noise variational component 
                        (MCsamp x Nnoise_R)
        """
        dispKey = self.priors['x1'].comp
        velKey = self.priors['x2'].comp
        rdispKey = self.priors['x3'].comp

        nData = len(base)
        (nSamp, nState) = samples[dispKey].shape
        bSize = int(nData/nState)

        tranSamp = {}
        for key in self.priors:
            p = self.priors[key]
            tranSamp[key] = p.invTransform(samples[p.comp][:,p.dim])

        states = torch.stack((samples[dispKey][:,:-1], samples[velKey][:,:-1], 
            samples[rdispKey][:,:-1]), dim=0)
        par = torch.stack((tranSamp['xi'], tranSamp['wn'],
                            tranSamp['beta'], tranSamp['n'], 
                            tranSamp['gamma']), dim=0)
        
        exc = base[:nData-bSize]

        for i in range(bSize):
            states = self.stochEuler(self.stochBW, states, par, exc[i::bSize]) 

        ## Transition Probability on X1 ##
        logX1= torch.distributions.Normal(states[0],                           \
                    tranSamp['w1']).log_prob(samples[dispKey][:,1:])
        
        ## Transition Probability on X2 ##
        logX2 = torch.distributions.Normal(states[1],                          \
                    tranSamp['w2']).log_prob(samples[velKey][:,1:])

        ## Transition Probability on X3 ##
        logX3 = torch.distributions.Normal(states[2],                          \
                    tranSamp['w3']).log_prob(samples[rdispKey][:,1:])
        
        if self.debug:
            print("Number of Data Points = \n" + str(nData) + "\n")
            print("Number of Samples = \n" + str(nSamp) + "\n")
            print("Number of States = \n" + str(nState) + "\n")
            print("Batch Size = \n" + str(bSize) + "\n")
            print("The shape of the input Matrix is \n" + str(base.shape) + "\n")
            print("The final displacement is \n" + str(states[0]) +"\n")
            print("The final velocity is \n" + str(states[1]) +"\n")
            print("The final Bouc-Wen displacement is \n" + str(states[2]) +"\n")
            print("Log of the Transition Probability on X1 = \n" + str(logX1) + "\n")
            print("Log of the Transition Probability on X2 = \n" + str(logX2) + "\n")
            print("Log of the Transition Probability on X3 = \n" + str(logX3) + "\n")
        
        return (torch.sum(logX1) + torch.sum(logX2) + torch.sum(logX3))/nSamp

    # def elbo(self, guide, data, base, nsamp): # Fully Defined by Parent Class

    ################################# Tools #################################
    def stochEuler(self, model, states, par, inp):
        return states + self.dt*model(states,par, exc=inp)

    def stochBW(self, states, par, exc=None):
        if exc is None:
            exc = torch.zeros(states[0].shape)

        y1 = states[1]
        y2 = (-exc - (2*par[0]*par[1])*states[1] 
                  - torch.pow(par[1],2)*states[2])
        y3 = (states[1] - par[2]*torch.abs(states[1])
            *torch.pow(torch.abs(states[2]), par[3]-1)*states[2]
            - par[4]*states[1]*torch.pow(torch.abs(states[2]), par[3]))

        return torch.stack((y1,y2,y3), dim=0)

    def __str__(self):
        return "Nonlinear energy sink stochastic time series model"

################################################################################
######################## Variational Inference Function ########################
################################################################################

def inferTimeSeries(model,guide,data,inp,lr, MC=1,nEpoch=1,epochIter=10000,    \
    sInt=100,rInt=0,anneal=False):
    """
    This function uses the classes defined previously, as well as the 
    variational family class, to implement ADVI over a specified number of 
    epochs and iterations. It returns the inferred means and standard deviations
    on the states and parameters, as well as the ELBO history. 

    model - StochasticTimeSeriesModels object describing the stochastic model of
            this particular dynamical system
    guide - dictonary of VariationalFamilies objects describing the full 
            variational family as a set of the variational components 
    data  - torch tensor of the observations of the system response
    inp   - torch tensor of the input to the system
    lr    - dictionary of learning rates, with keys matching the keys of the 
            inferred parameters in the guide
    MC    - number of MC samples to use in the gradient estimation
    nEpoch - number of epochs 
    epochIter - iterations of the stochastic gradient descent algorithm to use 
                in each epoch
    sInt  - interval at which to store data in the VariationalFamilies objects
    rInt  - interval at which to report out current ELBO to command prompt. Does
            not print if rInt = 0. 
    """

    if np.isscalar(epochIter):
        allIter = epochIter*nEpoch
        nStore = int(allIter/sInt + 1) # Number of values to store
    else: 
        allIter = np.sum(epochIter)
        nStore = int(allIter/sInt + 1) # Number of values to store
   
    ## Define Results Storage ##
    elboHist = np.zeros(nStore-1) 
    infMeans = {}
    infStds = {}
    for key in lr:
        if (key == 'disp') | (key == 'vel') | (key == 'rdisp'): # Store End States
            infMeans.update({key: np.zeros((nEpoch,guide[key].size))})  
            infStds.update({key: np.zeros((nEpoch,guide[key].size))}) 
        else:     # Store Iterations x Params
            infMeans.update({key: np.zeros((nStore, guide[key].size))}) 
            infMeans[key][0,:] = guide[key].vpHist[0,:guide[key].size]
            infStds.update({key: np.zeros((nStore, guide[key].size))})
            infStds[key][0,:] = guide[key].vpHist[0,guide[key].size:]
    
    runningIter = 0
    for j in range(nEpoch):
        print('\nEpoch #%d \n'%(j))

        opt = []
        if np.isscalar(epochIter):
            for key in lr:
                opt.append({'params':guide[key].vp, 'lr':lr[key]})
        else:
            for key in lr:
                opt.append({'params':guide[key].vp, 'lr':lr[key][j]})

        optimizer = torch.optim.Adam(opt)

        ## Stochastic Gradient Descent ##
        for i in range(epochIter[j]):
            # Clear gradients
            optimizer.zero_grad()

            # Calculate Annealing Factor
            runningIter += 1
            if anneal:
                aFactor = runningIter/(allIter)
            else:
                aFactor = 1
            

            # Calculate the Loss Function
            loss = -model.elbo(guide, data, inp, MC, aFactor)
            elboHolder = -loss.detach().numpy()

            ## Calculate the Gradient ##
            loss.backward()
            optimizer.step()

            ## Record Parameter History ##
            if (i + 1) % (sInt) == 0:
                if np.isscalar(epochIter):
                    elboHist[int(j*epochIter/sInt + (i+1)/sInt)-1] = elboHolder
                else: 
                    elboHist[int(np.sum(epochIter[:j])/sInt + (i+1)/sInt)-1] = elboHolder
                for key in guide:
                    guide[key].store(int((i+1)/sInt))

            ## Report Intermediate ELBOs ##
            if rInt == 0:
                pass
            else:
                if (i + 1) % (rInt) == 0:
                    now = datetime.datetime.now()
                    print('{} | {}/{} | elbo: {}'.format(now, i + 1, epochIter,
                        elboHolder))
        
        ## Store Intermediate Values ##
        for key in lr:
            if (key == 'disp') | (key == 'vel') | (key == 'rdisp'): # Store End States
                infMeans[key][j,:] = guide[key].vpHist[-1,:guide[key].size]
                holder = guide[key].buildMat().numpy()
                infStds[key][j,:] = np.sqrt(np.diag(np.matmul(holder,          \
                                    np.transpose(holder))))
            else:     # Store Iterations x Params
                if np.isscalar(epochIter):
                    infMeans[key][(1+int((epochIter/sInt)*j)):(1+int((epochIter/sInt)*(j+1))),:] =               \
                            guide[key].vpHist[1:,:guide[key].size]
                    infStds[key][(1+int((epochIter/sInt)*j)):(1+int((epochIter/sInt)*(j+1))),:] =                \
                            np.exp(guide[key].vpHist[1:,guide[key].size:])
                else: 
                    infMeans[key][(1+int(np.sum(epochIter[:j])/sInt)):(1+int(np.sum(epochIter[:j+1])/sInt)),:] =               \
                            guide[key].vpHist[1:,:guide[key].size]
                    infStds[key][(1+int(np.sum(epochIter[:j])/sInt)):(1+int(np.sum(epochIter[:j+1])/sInt)),:] =                \
                            np.exp(guide[key].vpHist[1:,guide[key].size:])

        ## Prep for the Next Iteration ##
        if j < (nEpoch-1):
            for key in guide:
                guide[key].reset(int(epochIter[j+1]/sInt))   
        
    return (elboHist, infMeans, infStds)