# -*- coding: utf-8 -*-
"""
Radioactivity Simulation Analysis v2.0

Lukas Kostal, 10.3.2023, ICL
"""


import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as so


# function for curve fitting straight line
def fit_lin(x, m, c):
    y = m * x + c
    
    return y


# function to calculate reduced chi squared value
def get_chi(E, O, O_err, param):
    chi = np.sum((O - E)**2 / O_err**2)
    chi = chi / (len(O) - param)
    
    return chi


# highest number in the name of the simulated data
file_max = 44

# simulated dataset to be analyzed
data_sim = 'Dataset_3'

# experimental dataset to be analyzed
data_exp = 'distance_redo'

# note significant portion of the experimental data analysis below is taken from Distance Analysis_3.py

# offset in distance and assicuated systematic error in m
dlt = 16.52e-3
dlt_err = 2e-3

# random error in distance measurement in m
d_err = 1e-3

# detector area in m^2
A_det = 1e-4

# background count rate and associated error in Bq
f_b = 0.1166
f_b_err = 0.0197

# load the data measured distance d in cm converted to m, time t in s, count n
d, t, n = np.loadtxt(f'Data/{data_exp}.txt', unpack=True, skiprows=1)
d *= 1e-2

# offset distance and associated random error
r = d + dlt
r_err = d_err

# lower and upper bounds on offset distance due to systematic error
r_p = r + dlt_err
r_m = r - dlt_err

# measured count rate and associated error in Bq
# note do not correct for dead time since it is included in G4 simulation
f = n / t - f_b
f_err = np.sqrt(n / t**2 + f_b_err**2)

# calculate adjusted count rate with random error
y = f * r**2
y_err = np.sqrt(r**2 * f_err**2 + 4*f**2 * r_err**2) * r

# lower and upper bounds on adjusted count rate due to systematic error
y_p = f * r_p**2
y_m = f * r_m**2

# load the parameters for the simulations distance d in cm and number of events N
r_sim, N = np.loadtxt(f'Simulation/{data_sim}/parameters.csv', unpack=True, delimiter=',', skiprows=1)
r_sim *= 1e-2

# slice the parameters for when not all macros are simulated
r_sim = r_sim[:file_max+1]
N = N[:file_max+1]

# array to hold number of detection events from simulations
n_sim = np.zeros(file_max+1)

# loop over each simulation output in the dataset
for i in range(1, file_max+1):
      
    # load the data from the simulation output
    data = np.loadtxt(f'Simulation/{data_sim}/data_{i}.txt', unpack=True, delimiter=',', skiprows=1)
    
    # get number of detection events
    n_sim[i] = len(data[0,:])

# remove the first simulated measurement since it has zero detections
r_sim = r_sim[1:]
N = N[1:]
n_sim = n_sim[1:]    

# fraction of simulated events detected together with statistical error
f_sim = n_sim / N
f_sim_err = np.sqrt(n_sim) / N

# adjusted fraction of simulated events detected and expected error
y_sim = f_sim * r_sim**2
y_sim_err = f_sim_err * y_sim / f_sim

#%%
# linear fit optimisaion onto the experimental dataset

# lists for fit parameters, error in fit, reduced chi squared number of measurements
i_arr = []
opt_arr = []
err_arr = []
chi_arr = []

# perform linear fit for increasing number of measuremens starting from 3
for i in range(3, len(y)):
    
    # perform linear fit, get errors from ECM, convert fit parameters to function and get reduced chi squared
    opt, cov = so.curve_fit(fit_lin, r[-i:], y[-i:], sigma=None, absolute_sigma=True)
    err = np.sqrt(np.diagonal(cov))
    lin = np.poly1d(opt)
    chi = get_chi(lin(r[-i:]), y[-i:], y_err[-i:], 2)
    
    # append calculated values to the lists
    i_arr.append(i)
    opt_arr.append(opt)
    err_arr.append(err)
    chi_arr.append(chi)

# convert lists of arrays to 2D arrays
i_arr = np.asarray(i_arr)    
opt_arr = np.asarray(opt_arr)
err_arr = np.asarray(err_arr)
chi_arr = np.asarray(chi_arr)

# find the number of measurements which minimises error in fit
i_min = np.argmin(err_arr[:, 0])

# perform curve fit to get line of best fit
opt, cov = so.curve_fit(fit_lin, r[-(i_min):], y[-(i_min):], sigma=y_err[-(i_min):], absolute_sigma=True)

# convert coefficients from curve fit into function
lin = np.poly1d(opt)

# get the reduced chi squared value for the linear fit
chi = get_chi(lin(r[-i_min:]), y[-i_min:], y_err[-i_min:], 2)

# perform the linear fit using the upper and lower bounds on y
opt_p, cov_p = so.curve_fit(fit_lin, r[-(i_min):], y_p[-(i_min):], sigma=y_err[-(i_min):], absolute_sigma=True)
opt_m, cov_m = so.curve_fit(fit_lin, r[-(i_min):], y_m[-(i_min):], sigma=y_err[-(i_min):], absolute_sigma=True)

# find the error in the fit for the upper and lower bounds
err_p = np.sqrt(np.diagonal(cov_p))
err_m = np.sqrt(np.diagonal(cov_m))

# generate functions for fit from upper and lower bounds
# this time include systeamtic error as well as fit error
lin_p = np.poly1d(opt_p + err_p)
lin_m = np.poly1d(opt_m - err_m)

#%%
# linear fit optimisation onto the simulation dataset

# list to store fit parameters this time for the simulation dataset
i_sim_arr = []
opt_sim_arr = []
err_sim_arr = []
chi_sim_arr = []

# perform linear fit for increasing number of simulated measurements starting with 3
for i in range(3, len(y)):

    # perform linear fit, get errors from ECM, convert fit parameters to function and get reduced chi squared
    opt_sim, cov_sim = so.curve_fit(fit_lin, r_sim[-i:], y_sim[-i:], sigma=None, absolute_sigma=False)
    err_sim = np.sqrt(np.diagonal(cov_sim))
    lin_sim = np.poly1d(opt_sim)
    chi_sim = get_chi(lin(r_sim[-i:]), y_sim[-i:], y_sim_err[-i:], 2)
    
    # append calculated values to the lists
    i_sim_arr.append(i)
    opt_sim_arr.append(opt_sim)
    err_sim_arr.append(err_sim)
    chi_sim_arr.append(chi_sim)

# convert lists of arrays to 2D arrays
i_sim_arr = np.asarray(i_sim_arr)    
opt_sim_arr = np.asarray(opt_sim_arr)
err_sim_arr = np.asarray(err_sim_arr)
chi_sim_arr = np.asarray(chi_sim_arr)

# find the number of measurements which minimises error in fit
i_sim_min = np.argmin(err_sim_arr[:, 0])

# perform curve fit to get line of best fit for simulated dataset
opt_sim, cov_sim = so.curve_fit(fit_lin, r_sim[-(i_sim_min):], y_sim[-(i_sim_min):], sigma=y_sim_err[-(i_sim_min):], absolute_sigma=True)

#%%
# re-normalising the simulation dataset to match the effective activity of source used in experiment

# get scaling factor from ratio of y intercepts
# this ensures the simulated dataset represents source of same effective activity as experimental data
S = (opt / opt_sim)[1]

# apply the scaling to the simulation adjusted count rate and associated error
y_sim *= S
y_sim_err *= S

# perform curve fit to get line of best fit for simulated dataset after re-normalisation scaling
opt_sim, cov_sim = so.curve_fit(fit_lin, r_sim[-(i_sim_min):], y_sim[-(i_sim_min):], sigma=y_sim_err[-(i_sim_min):], absolute_sigma=True)

# convert fit parameters into function
lin_sim = np.poly1d(opt_sim)

# get the reduced chi squared value for simulated dataset
chi_sim = get_chi(lin_sim(r_sim[-i_sim_min:]), y_sim[-i_sim_min:], y_sim_err[-i_sim_min:], 2)

# get error from ECM
err_sim = np.sqrt(np.diagonal(cov_sim))

# generate functions for lines representing errors in simulation dataset fitting
lin_sim_p = np.poly1d(opt_sim + err_sim)
lin_sim_m = np.poly1d(opt_sim - err_sim)

#%%

# print the numerical outputs
print()
print(f'reduced chi squared for experimental data  = {chi:.4g}')
print(f'reduced chi squared for simulated data     = {chi_sim:.4g}')

#%%

# start plotting
fig, ax = plt.subplots(1, 1)

# plot adjusted count rate from simulation and experimental datasets
ax.plot(r_sim, y_sim, ls='none', marker='.', color='blue', zorder=10)
ax.plot(r, y, ls='none', marker='.', color='green', zorder=9)

# plot opaque line between the points just to make the shape of curve easier to see
ax.plot(r_sim, y_sim, ls='-', color='blue', alpha=0.2, zorder=8)
ax.plot(r, y, ls='-', color='green', alpha=0.2, zorder=7)

# plot errorbars for the adjusted count rate of both datasets
ax.errorbar(r_sim, y_sim, yerr = y_sim_err, fmt='.', color='royalblue', zorder=4, label='G4 simulation')
ax.errorbar(r, y, xerr=r_err, yerr=y_err, fmt='.', color='limegreen', zorder=3, label='measurement')

# plot fitted lines
ax.plot(r_sim, lin_sim(r_sim), color='cyan', zorder=6, label='G4 simulation lin fit')
ax.plot(r, lin(r), color='orange', zorder=5, label='measurement lin fit')

# plot the regions representing the total error in the linear fits
ax.fill_between(r_sim, lin_sim_p(r_sim), lin_sim_m(r_sim), color='cyan', alpha=0.2, zorder=2, label='G4 simulation lin error')
ax.fill_between(r, lin_p(r), lin_m(r), color='orange', alpha=0.2, zorder=1, label='measurement lin error')

# title and labels for plotting
plt.title('Experimental Data and Geant4 Simulation')
plt.xlabel('distance $r$ ($m$)')
plt.ylabel(r'adjusted count rate $\frac{n}{\Delta t} r^2$ ($Bq \; m^2$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()
handles, labels = plt.gca().get_legend_handles_labels()
order = [5, 4, 0, 1, 2, 3]
plt.legend([handles[i] for i in order], [labels[i] for i in order], markerscale=1.2)

# set axis range to make curves more vissible
plt.xlim(0, 0.51)
plt.ylim(10, 22)

# save the plot
plt.savefig('Plots/Simulation/simulation_adjusted.png', dpi=300, bbox_inches='tight')
plt.show()
