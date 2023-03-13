# -*- coding: utf-8 -*-
"""
Radioactivity Thickness Analysis v3.0

Lukas Kostal, 8.3.2023, ICL
"""


import numpy as np
import scipy.constants as sc
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy.optimize as sco


# function for curve fitting straight line
def fit_lin(x, m, c):
    y = m * x + c
    
    return y


# function to calculate reduced chi squared value
def get_chi(E, O, O_err, param):
    chi = np.sum((O - E)**2 / O_err**2)
    chi = chi / (len(O) - param)
    
    return chi

# function to calcualte x coordinate of intercept of two lines with error
def get_itc(opt_1, opt_2, err_1, err_2):
    x = - (opt_1 - opt_2)[1] / (opt_1 - opt_2)[0]
    
    err = err_1**2 + err_2**2
    x_err = np.sqrt(err[1] + x**2 * err[0]) / (opt_1 - opt_2)[0]
    x_err = np.abs(x_err)
    
    return x, x_err


# function to generate 3 line function from fit parameters and errors
def get_lins(opt, err):
    lin = np.poly1d(opt)
    lin_p = np.poly1d(opt + err)
    lin_m = np.poly1d(opt - err)
    
    return lin, lin_p, lin_m


# dataset to be analysed
dataset = 'Task_20_redo'

# specify material being analysed
mat = 'Cu'

# specify whether to choose measurements for line fitting automatically
choose_measurements = 'auto'

# background count rate and associated error in Bq
fb = 0.1166
fb_err = 0.0197

# error in measurement of thickness in mm
h_err = 0.01

# load data thickness h in mm, time interval t in s, number of counts c
h, t, n = np.loadtxt(f'Data/{dataset}.txt', unpack=True, skiprows=1)

# convert thickness and error to m
h *= 1e-3
h_err *= 1e-3

# calculate the count rate and expected uncertainty
f = n / t - fb
f_err = np.sqrt(n / t**2 + fb_err**2)

# calculate the log count rate and expected uncertainty
f_log = np.log(f_err) / np.log(10)
f_log_err = f_err / f / np.log(10)

# add error in thickness and log count rate to get combined error for fitting
err_quad = np.sqrt(h_err**2 + f_log_err**2)

# array of indexes for the dataset
j_arr = np.arange(0, len(f))

# density and indexes of mesaurements for fitting for Al material
if mat == 'Al':
    # density and error in kg m^-3
    rho = 2705
    rho_err = 0
    
    # arrays of indexes to group measurements into 3 sets and fit lines
    j_1 = j_arr[:14]
    j_2 = j_arr[13:20]
    j_3 = j_arr[20:]
    
# same as before but now for Cu
if mat == 'Cu':
    rho = 8960
    rho_err = 0

    j_1 = j_arr[:7]
    j_2 = j_arr[6:11]
    j_3 = j_arr[12:]
    
# automatically find indexes to group measurements by minimisising reduced chi squared
if choose_measurements == 'auto':
    
    # list to hold chi squared for each permutation of grouping measurements
    chi_arr = []
    
    # list to hold the index of measurement at which they are split
    k_arr = []
    
    # loop over all possible points for first and second split k_1 and k_2 respectively
    for k_1 in range(3, len(f)-6):
        for k_2 in range(k_1+3, len(f)-3):
            
            # chi squared value for the current grouping of measurements
            chi=0
            
            # loop over the 3 groups of measurements performing lin fit for each
            for i in range(0, 3):
                
                # get arrays of indexes for the current grouping of measurements
                if i == 0:
                    j = j_arr[:k_1]
                if i == 1:
                    j = j_arr[k_1:k_2]
                if i == 2:
                    j = j_arr[k_2:]
            
                # perform curve fit of straight line onto the set of measurements
                opt, cov = sco.curve_fit(fit_lin, h[j], f_log[j], sigma=err_quad[j], absolute_sigma=True)
                
                # covert the coefficeints from linear fit to function
                lin = np.poly1d(opt)
                
                # calcualte contribution of the set of points to total chi squared value
                chi += np.sum((f_log[j] - lin(h[j]))**2 / err_quad[j]**2)
            
            # append the reduced chi squared values and indexes where groups split
            chi_arr.append(chi)
            k_arr.append([k_1, k_2])
    
    # convert the lists into numpy arrays
    chi_arr = np.array(chi_arr)
    k_arr = np.array(k_arr)
    
    # divide by DOF to get reduced chi squared
    chi_arr /= len(f) - 6
    
    # find the index at which reduced chi squared is minimised
    j_min = np.argmin(chi_arr)
    
    # get the indexes at which the groups split for minimised chi squared
    k_min = k_arr[j_min]
    
    # arrays of indexes of measurements for grouping which minimised chi squared
    j_1 = j_arr[:k_min[0]]
    j_2 = j_arr[k_min[0]:k_min[1]]
    j_3 = j_arr[k_min[1]:]
    
    # plot contour plot of chi squared against grouping of measurements
    lvl = np.linspace(np.amin(chi_arr), np.amax(chi_arr), 20)
    plt.tricontourf(k_arr[:,0], k_arr[:,1], chi_arr, cmap='plasma', levels=lvl)
    plt.colorbar(label=r'reduced $\chi^2$ $(unitless)$')
    
    # plot point at which reduced chi squared is minimised
    plt.plot(k_min[0], k_min[1], 'x', color='red')
    plt.axvline(x=k_min[0], ls='--', color='red')
    plt.axhline(y=k_min[1], ls='--', color='red')
    
    # title and labels for plotting
    plt.title('Reduced $\chi^2$ against Measurement Grouping')
    plt.xlabel('no of measurements in lin fit 1')
    plt.ylabel('no of measurements in lin fit 2')
    plt.rc('grid', linestyle=':', color='black', alpha=0.8)
    plt.grid()
    plt.axis('equal')

    # save the plot
    plt.savefig(f'Plots/Thickness/{dataset}_contour.png', dpi=300, bbox_inches='tight')
    plt.show()

    # plot the reduced chi squared array
    plt.plot(chi_arr, color='blue')
    
    # plot point at which reduced chi squared is minimised
    plt.plot(j_min,chi_arr[j_min],'x', color='red')
    plt.axvline(x=j_min, ls='--', color='red')
    plt.axhline(y=chi_arr[j_min], ls='--', color='red')
    
    # title and labels for plotting
    plt.title('Concatenated Reduced $\chi^2$ Array')
    plt.xlabel('array index $i$')
    plt.ylabel('reduced $\chi^2$ $(\log(Bq))$')
    plt.rc('grid', linestyle=':', color='black', alpha=0.8)
    plt.grid()

    # save the plot
    plt.savefig(f'Plots/Thickness/{dataset}_array.png', dpi=300, bbox_inches='tight')
    plt.show()

# array of arrays for seting the measurements
j_arr = np.array([j_1, j_2, j_3], dtype=object)

# prepare empty arrays to hold arrays from analysis of each set of measurements
opt = np.empty(3, dtype=object)
cov = np.empty(3, dtype=object)
err = np.empty(3, dtype=object)
chi = np.empty(3)

chi_tot = 0

# loop over all 3 sets of measurements
for i in range(0, 3):
    j = j_arr[i]
    
    # perform curve fit of straight line onto the set of measurements
    opt[i], cov[i] = sco.curve_fit(fit_lin, h[j], f_log[j], sigma=err_quad[j], absolute_sigma=True)

    # get error in curve fit parameters from error covariance matrix
    err[i] = np.sqrt(np.diagonal(cov[i]))
    
    # covert the coefficeints from linear fit to function
    lin = np.poly1d(opt[i])
    
    # calcualte separate reduced chi squared value for each line of best fit
    chi[i] = get_chi(lin(h[j]), f_log[j], err_quad[j], 2)
    
    # calcualte contribution of the set of points to total chi squared value
    chi_tot += np.sum((f_log[j] - lin(h[j]))**2 / err_quad[j]**2)
    
# divide by degrees of freedom for all 3 lines to get reduced chi squared
chi_tot /= len(f) - 6

# calcualte thickness at intersection of lines and associated error
itc_1, itc_1_err = get_itc(opt[0], opt[1], err[0], err[1])
itc_2, itc_2_err = get_itc(opt[1], opt[2], err[1], err[2])

# calcualte range of the beta particles and associated error in kg m^-2
R = itc_2 * rho
R_err = R * np.sqrt((itc_2_err / itc_2)**2 + (rho_err / rho)**2)

# calcualte the maximum kinetic energy and associated error in MeV
E = np.sqrt(5/112 * ((10/11 * R + 1)**2 - 1))
E_err = 25/616 * (10/11 * R + 1) / E * R_err

# output numerical results
print()
print(f'Chi_1       = {chi[0]:.4g}')
print(f'Chi_2       = {chi[1]:.4g}')
print(f'Chi_3       = {chi[2]:.4g}')
print(f'Chi_total   = {chi_tot:.4g}')
print()
print(f'h_1         = {itc_1:.4g} ± {itc_1_err:.4g} mm')
print(f'h_2         = {itc_2:.4g} ± {itc_2_err:.4g}mm')
print()
print(f'material    : {mat}')
print(f'density     = {rho} kg m^-3')
print(f'R           = {R:.4g} kg m^-2')
print(f'E_max       = {E:.4g} \u00B1 {E_err:.4g} MeV')
print()

# arrays of thickness for plotting each line
h_1 = np.linspace(np.amin(h), itc_1, 100)
h_2 = np.linspace(itc_1, itc_2, 100)
h_3 = np.linspace(itc_2, np.amax(h), 100)

# array of arrays of thickness for plotting within a loop
h_arr = np.array([h_1, h_2, h_3], dtype=object)

# array of colors for plotting within a loop
color_arr = ['green', 'blue', 'red']

# plot the measurements with errors note thickness on plots is in mm
plt.errorbar(h*1e3, f_log, xerr=h_err*1e3, yerr=f_log_err, ls='none', marker='.', markersize=4, capsize=3, color='black', label='measurement')

# loop to plot the linear fits
for i in range(0, 3):
    # get functions of lines from coefficients and errors
    lin, lin_p, lin_m = get_lins(opt[i], err[i])
    
    # convert thickness to list and back to array to prevent dtype error
    h_plt = np.array(h_arr[i].tolist())
    
    # plot lines to indicate measurement groupings
    plt.axvline(x=h[j_arr[i][0]]*1e3, ls='--', color=color_arr[i])
    plt.axvline(x=h[j_arr[i][-1]]*1e3, ls='--', color=color_arr[i])
    
    # plot the line of best fit
    plt.plot(h_plt*1e3, lin(h_plt), color=color_arr[i], label=f'lin fit {i+1}')
    
    # plot the region corresponding to total error in linear fit
    plt.fill_between(h_plt*1e3, lin_p(h_plt), lin_m(h_plt), color=color_arr[i], alpha=0.2)

# title and labels for plotting
plt.title(f'Log Count Rate against {mat} Thickness')
plt.xlabel('thickness $h$ ($mm$)')
plt.ylabel('log count rate $\log(f)$ ($\log(Bq)$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()
plt.legend()

# save the plot
plt.savefig(f'Plots/Thickness/{dataset}.png', dpi=300, bbox_inches='tight')
plt.show()