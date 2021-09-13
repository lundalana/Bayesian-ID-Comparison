"""
File Name: inp.py
Purpose: This file contains a series of functions used to generate input 
	  excitations for dynamical systems models. A plotting function is 
	  also included that has the option to be called from within each of 
	  the input generation functions. 
Developed by: Alana Lund (Purdue University) 
Last Updated: 13 Sept. 2021 
License: AGPL-3.0
""" 

import numpy as np
import scipy.signal as sigproc
import math
import matplotlib.pyplot as plt
import seaborn

def sweep(tMax, fMax, aMax, fs, plot=False, fSize = (16, 10)):
	"""
	This function produces a displacement sine sweep, with linearly increasing 
	frequency. 

	tMax = maximum time (scalar)
	fMax = maximum frequency (scalar)
	aMax = amplitude (scalar)
	fs = sampling frequency (scalar)
	plot = plot the results? (binary)
	fSize = size of figures generated (tuple)
	"""
	dt = 1./fs # step size
	time = np.arange(0,tMax, dt)
	rate = fMax/tMax    # rate at which the frequency increases
	freq = rate*time/2. # frequency at any given time

	sweep = np.multiply(freq,time)
	inpDisp = aMax*np.sin(2*math.pi*sweep)
	inpAcc = 4.*math.pi*aMax*rate*(np.cos(2*math.pi*sweep) - 
	            4*math.pi*np.multiply(sweep,np.sin(2*math.pi*sweep)))

	if plot:
		plotInp(time, fMax, disp = inpDisp, acc = inpAcc, fSize = fSize)

	return (time, inpDisp, inpAcc)

def sine(tMax, fMax, aMax, fs, plot=False, fSize = (16, 10)):
	"""
	This function produces a displacement sine wave, with linear increasing 
	amplitude. 

	tMax = maximum time (scalar)
	fMax = frequency (scalar)
	aMax = amplitude (scalar)
	fs = sampling frequency (scalar)
	plot = plot the results? (binary)
	fSize = size of figures generated (tuple)
	"""
	dt = 1./fs # step size
	time = np.arange(0,tMax, dt)
	amprate = aMax/tMax    # rate at which the frequency increases
	amp = amprate*time     # amplitude at any given time

	inpDisp = np.multiply(amp,np.sin(2*math.pi*fMax*time))
	inpAcc = (4.*math.pi*fMax*amprate*np.cos(2*math.pi*fMax*time)- 
			(2*math.pi*fMax)**2*np.multiply(amp,np.sin(2*math.pi*fMax*time)))

	if plot:
		plotInp(time, fMax, disp = inpDisp, acc = inpAcc, fSize = fSize)

	return (time, inpDisp, inpAcc)

def free(tMax, fMax, aMax, fs, plot=False, fSize = (16, 10)):
	"""
	This function produces a zero displacement signal for free response.
	Note that for a result to be produced, initial conditions must then 
	be set to non-zero.  

	tMax = maximum time (scalar)
	fMax = maximum frequency (scalar)
	aMax = amplitude (scalar)
	fs = sampling frequency (scalar)
	plot = plot the results? (binary)
	fSize = size of figures generated (tuple)
	"""
	dt = 1./fs # step size
	time = np.arange(0,tMax, dt)
	inpDisp = np.zeros(len(time))
	inpAcc = np.zeros(len(time))

	if plot:
		plotInp(time, fMax, disp = inpDisp, acc = inpAcc, fSize = fSize)

	return (time, inpDisp, inpAcc)

def BLWN(tMax, fMax, aMax, fs, plot=False, fSize = (16, 10)):
	"""
	This function produces a BLWN signal in acceleration. 

	tMax = maximum time (scalar)
	fMax = maximum frequency (scalar)
	aMax = amplitude (scalar)
	fs = sampling frequency (scalar)
	plot = plot the results? (binary)
	fSize = size of figures generated (tuple)
	"""
	### Specify Pass-Band ###
	fMin = 0.  # Minimum desired frequency in Hz
	freqs = np.abs(np.fft.fftfreq(int(fs*tMax), 1/fs))
	f = np.zeros(int(fs*tMax))
	idx = np.where(np.logical_and(freqs>=fMin, freqs<=fMax))[0]
	f[idx] = 1

	### Generates BLWN Sequence ###
	f = np.array(f, dtype='complex')
	Np = (len(f) - 1) // 2
	phases = np.random.rand(Np) * 2 * np.pi
	phases = np.cos(phases) + 1j * np.sin(phases)
	f[1:Np+1] *= phases
	f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
	inpAcc =  np.fft.ifft(f).real

	### Scale to Maximum Amplitude ###
	inpAcc = aMax*inpAcc/np.amax(np.abs(inpAcc))
	time = np.arange(len(inpAcc))/fs

	if plot:
		plotInp(time, fMax, disp = np.zeros(len(time)), acc = inpAcc, 
			fSize = fSize)

	return (time, np.zeros(len(time)), inpAcc)


def plotInp(time, fMax, disp = 0, acc = 0,  fSize = (16, 10)):
	"""
	This function plots the input signals generated in this file. 

	time = time increments over which the signal operates (vector)
	disp = displacement history (vector)
	acc = acceleration history (vector)
	fs = sampling frequency (scalar)
	fSize = size of figures generated (tuple)
	"""
	## Format Inputs ##
	if acc.size == 1:
		acc = np.zeros(len(time))
	if disp.size == 1:
		disp = np.zeros(len(time))

	## Plot Displacement ##
	if (disp[-1] != 0):
		fig, ax = plt.subplots(1,2, figsize = fSize)

		# Time History #
		ax[0].plot(time, disp)
		ax[0].set_xlabel('Time [sec]')
		ax[0].set_ylabel(r'Disp. [m]')
		ax[0].set_title('Base Displacement')
		ax[0].set_xlim((0, time[-1]))
		ax[0].grid()

		# Amplitude Density Spectrum #
		fs = 1/(time[1]-time[0])                            # sampling frequency
		NFFT = int(2**(np.ceil(np.log2(np.abs(len(disp))))+4))
		yInp = np.fft.fft(disp, n=NFFT)/len(disp)
		freq = fs*np.linspace(0,1,int(NFFT))

		ax[1].plot(freq[0:int(NFFT/2)], 2*np.abs(yInp[0:int(NFFT/2)]))
		ax[1].set_xlabel('Freq [Hz]')
		ax[1].set_ylabel(r'Mag. [m/Hz]')
		ax[1].set_title('Single-Sided Amplitude Density Spectrum')
		ax[1].set_xlim((0, fMax*2))
		ax[1].grid(1)

		plt.tight_layout()

	## Plot Acceleration ##
	if (acc[-1] != 0):
		fig, ax = plt.subplots(1,2, figsize = fSize)

		# Time History #
		ax[0].plot(time, acc)
		ax[0].set_xlabel('Time [sec]')
		ax[0].set_ylabel(r'Acc. [$\mathrm{m/sec}^2$]')
		ax[0].set_title('Base Acceleration')
		ax[0].set_xlim((0, time[-1]))
		ax[0].grid()

		# Amplitude Density Spectrum #
		fs = 1/(time[1]-time[0])                            # sampling frequency
		NFFT = int(2**(np.ceil(np.log2(np.abs(len(acc))))+4))
		yInp = np.fft.fft(acc, n=NFFT)/len(acc)
		freq = fs*np.linspace(0,1,int(NFFT))

		ax[1].plot(freq[0:int(NFFT/2)], 2*np.abs(yInp[0:int(NFFT/2)]))
		ax[1].set_xlabel('Freq [Hz]')
		ax[1].set_ylabel(r'Mag. [($\mathrm{m/sec}^2$)/Hz]')
		ax[1].set_title('Single-Sided Amplitude Density Spectrum')
		ax[1].set_xlim((0, fMax*2))
		ax[1].grid(1)

		plt.tight_layout()

	return
