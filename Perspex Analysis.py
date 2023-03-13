# -*- coding: utf-8 -*-
"""
Radioactivity Perspex Analysis v1.0

Lukas Kostal, 9.3.2023, ICL
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate as si
import scipy.optimize as so


# function for trapezium numerical integration
def num_int(x, y):
    w = np.diff(x)
    A = 0
    for i in range(0, len(w)):
        A += w[i] * (y[i] + y[i+1])/2

    return A


# uncorrected measured activity of the source in MBq
Act = 2.1628
Act_err_p = 0.0849
Act_err_m = 0.0826

# thickenss of Perspex layer and associated error in m
h = 0.97e-3
h_err = 0.02e-3

# density of Perspex layer and associated error in m
rho = 1190
rho_err = 10

# name of data file for stopping power of Perspex
stop_pow = 'ESTAR_Perspex'

# name of data files for energy spectrum of Sr90 and Y90
Sr_spec = 'Sr90/beta-_Sr90_tot'
Y_spec = 'Y90/beta-_Y90_tot'

# load the data for the stopping power of Perspex
# stopping energy in MeV, mass stopping powers in MeV cm^2 g^-1 and density effect parameter
E_stop, col, rad, n_stop, dens = np.loadtxt(f'Perspex/{stop_pow}.txt', unpack=True, delimiter=' ', skiprows=8)

# convert energy into keV and mass stopping power into keV cm^2 g^-1
E_stop *= 1e3
n_stop *= 1e3 / 10

# load the data for energy spectrum of Sr90 and Y90 in keV
E_Sr, n_Sr, err_Sr = np.loadtxt(f'Perspex/{Sr_spec}.csv', unpack=True, delimiter=',', skiprows=1)
E_Y, n_Y, err_Y = np.loadtxt(f'Perspex/{Y_spec}.csv', unpack=True, delimiter=',', skiprows=1)

# fit cubic spline onto the energy spectra to combine them
cs_Sr = si.CubicSpline(E_Sr, n_Sr)
cs_Y = si.CubicSpline(E_Y, n_Y)

# array of energies in keV for the combined spectrum
E_spec = np.linspace(0, np.amax(E_Y), 10000)
n_spec = np.zeros(10000)

# combine the two energy spectra
for i in range(0, 10000):
    
    if E_spec[i] < np.amax(E_Sr):
        n_spec[i] = cs_Sr(E_spec[i]) + cs_Y(E_spec[i])
    else:
        n_spec[i] = cs_Y(E_spec[i])

# integrate over the combined spectrum and normalise it
N_tot = num_int(E_spec, n_spec)
n_spec /= N_tot

# arrays to hold stopping distance and upper and lower bounds for error propagation
h_stop = np.zeros(len(E_stop))
h_stop_p = np.zeros(len(E_stop))
h_stop_m = np.zeros(len(E_stop))

# loop to integrate up to increasing maximum energy to find stopping distance at that energy
for i in range(1, len(E_stop)):
    # calcualte the reciprocal of the stopping power and integrate
    n_int = 1 / (n_stop[:i] * rho)
    h_stop[i] = num_int(E_stop[:i], n_int)
    
    # repeat for upper and lower bounds on density from systeamtic error
    n_int_p = 1 / (n_stop[:i] * (rho + rho_err))
    n_int_m = 1 / (n_stop[:i] * (rho - rho_err))
    h_stop_p[i] = num_int(E_stop[:i], n_int_p)
    h_stop_m[i] = num_int(E_stop[:i], n_int_m)

# cubic spline fit onto the calcualted stopping distance and the upper and lower bounds
cs_stop = si.CubicSpline(E_stop, h_stop - h)
cs_stop_p = si.CubicSpline(E_stop, h_stop_p - (h + h_err))
cs_stop_m = si.CubicSpline(E_stop, h_stop_m - (h - h_err))

# use Newton-Raphson method to find the maximum energy of particles which are stopped by 1mm perspex
E_max = so.newton(cs_stop, 200, maxiter=500)
E_max_p = so.newton(cs_stop_p, 200, maxiter=500)
E_max_m = so.newton(cs_stop_m, 200, maxiter=500)

# array of energies from energy spectrum which are below the calcualted cutoff energies
E = E_spec[(E_spec < E_max)]
E_p = E_spec[(E_spec < E_max_p)]
E_m = E_spec[(E_spec < E_max_m)]

# array of probability densities from energy spectrum which are below the cutoff energies
n = n_spec[(E_spec < E_max)]
n_p = n_spec[(E_spec < E_max_p)]
n_m = n_spec[(E_spec < E_max_m)]

# integrate the combined energy spectrum up to the determined maximum energy value
# this calculates the fraction of electrons emitted by the source which are stopped by the perspex sheet
f = num_int(E, n)
f_p = num_int(E_p, n_p)
f_m = num_int(E_m, n_m)

# find the corrected activity of the source
cAct = Act / (1 - f)

# find the upper and lower bounds on the corrected activity due to error in original value and correction
cAct_p = (Act + Act_err_p) / (1 - f_m)
cAct_m = (Act - Act_err_m) / (1 - f_p)

# find errors for numerical output from the highest and lowest bounds calcualted throught
cAct_err_p = np.abs(cAct - cAct_p)
cAct_err_m = np.abs(cAct - cAct_m)
E_max_err_p = np.abs(E_max - E_max_p)
E_max_err_m = np.abs(E_max - E_max_m)
f_err_p = np.abs(f - f_p)
f_err_m = np.abs(f - f_m)

# print the numerical results
print()
print(f'E_max           = {E_max:.4g} +{E_max_err_p:.4g} -{E_max_err_m:.4g} keV')
print()
print(f'f_stop          = {f:.4g} +{f_err_p:.4g} -{f_err_m:.4g}')
print(f'f_pass          = {(1-f):.4g} +{f_err_m:.4g} -{f_err_p:.4g}')
print()
print(f'A_measured      = {Act:.4g} +{Act_err_p} -{Act_err_m} MBq')
print(f'A_corrected     = {cAct:.4g} +{cAct_err_p:.4g} -{cAct_err_m:.4g} MBq')

# plot mass stopping power against electron energy
plt.plot(E_stop, n_stop, color='blue')

# title and labels for plotting
plt.title(r'Perspex Mass Stopping Power against $\beta$ Energy')
plt.xlabel('energy $E$ ($keV$)')
plt.ylabel(r'mass stopping power ($keV \; m^2 \; kg^{-1}$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

# save the plot
plt.savefig('Plots/Perspex/PerspexStoppingPower.png', dpi=300, bbox_inches='tight')
plt.show()

# plot density effect parameter
plt.plot(E_stop, dens)

# title and labels for plotting
plt.title('Perspex Density Effect Parameter against $\\beta$ Energy')
plt.xlabel('energy $E$ ($keV$)')
plt.ylabel(r'desnity effect parameter $\delta \rho$ ($unitless$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

# save the plot
plt.savefig('Plots/Perspex/DensityEffect.png', dpi=300, bbox_inches='tight')
plt.show()

# plot maximum energy stopped against perspex thickness
plt.plot(h_stop * 1e3, E_stop, color='blue')
plt.axvline(x=(h - h_err)*1e3, ls='--', color='red')
plt.axvline(x=(h + h_err)*1e3, ls='--', color='red')
plt.axhline(y=E_max_p, ls='--', color='red')
plt.axhline(y=E_max_m, ls='--', color='red')

# title and labels for plotting
plt.title(r'Maximum Energy of Stopped $\beta$')
plt.xlabel(r'thickness $h$ ($mm$)')
plt.ylabel(r'maximum energy stoppep $E_{max}}$ ($keV$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

# save the plot
plt.savefig('Plots/Perspex/StoppingThickness.png', dpi=300, bbox_inches='tight')
plt.show()

# plot the energy spectrum of the Sr90 source
plt.plot(E_Sr, n_Sr, color='limegreen', label='Sr 90')
plt.plot(E_Y, n_Y, color='orange', label='Y 90')
plt.plot(E_spec, n_spec, color='blue', label='combined')
plt.axvline(x=E_max_p, ls='--', color='red')
plt.axvline(x=E_max_m, ls='--', color='red', label=r'E_max of stopped $\beta$')

# title and labels for plotting
plt.title(r'Sr90 Source Energy Spectrum')
plt.xlabel(r'energy $E$ ($keV$)')
plt.ylabel(r'probability density $\frac{dN}{dE}$ ($keV^{-1}$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()
plt.legend()

# save the plot
plt.savefig('Plots/Perspex/EnergySpectrum.png', dpi=300, bbox_inches='tight')
plt.show()

