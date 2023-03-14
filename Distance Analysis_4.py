# -*- coding: utf-8 -*-
"""
Radioactivity Distance Analysis v4.2

Lukas Kostal, 12.3.2023, ICL
"""


import numpy as np
import scipy.constants as sc
from matplotlib import pyplot as plt
import scipy.optimize as so
import scipy.interpolate as si


# function for curve fitting straight line
def fit_lin(x, m, c):
    y = m * x + c
    
    return y


# function to calculate reduced chi squared value
def get_chi(E, O, O_err, param):
    chi = np.sum((O - E)**2 / O_err**2)
    chi = chi / (len(O) - param)
    
    return chi


# dataset to be analysed
dataset = 'distance_redo'

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

# dead time of the detector and associated error in s
t_d = 1.6e-6
t_d_err = 3e-7

# load the data measured distance d in cm converted to m, time t in s, count n
d, t, n = np.loadtxt(f'Data/{dataset}.txt', unpack=True, skiprows=1)
d *= 1e-2

# offset distance and associated random error
r = d + dlt
r_err = d_err

# lower and upper bounds on offset distance due to systematic error
r_p = r + dlt_err
r_m = r - dlt_err

# calcualted measured count rate and subtract background in Bq
f_d = n / t - f_b
f_err_d = np.sqrt(n / t**2 + f_b_err**2)

# correct for dead time to get actual count rate in Bq and propagate random error
f = f_d * np.exp(t_d * f_d)
f_err = np.exp(t_d * f_d) * np.sqrt((1 + f_d * t_d)**2 * f_err_d**2)

# lower and upper bounds for actual count rate due to systeamtic error in count rate
f_p = f_d * np.exp((t_d + t_d_err) * f_d)
f_m = f_d * np.exp((t_d - t_d_err) * f_d)

# calculate adjusted count rate with random error
y = f * r**2
y_err = np.sqrt(r**2 * f_err**2 + 4*f**2 * r_err**2) * r

# lower and upper bounds on adjusted count rate due to systematic error
y_p = f_p * r_p**2
y_m = f_m * r_m**2

# lists for fit parameters, error in fit, reduced chi squared number of measurements
i_arr = []
opt_arr = []
err_arr = []
chi_arr = []

# perform linear fit for increasing number of measuremens starting from 3
for i in range(3, len(y)):
    # opt, cov = np.polyfit(r[-i:], y[-i:], 1, cov=True)
    opt, cov = so.curve_fit(fit_lin, r[-i:], y[-i:], sigma=None, absolute_sigma=True)
    err = np.sqrt(np.diagonal(cov))
    lin = np.poly1d(opt)
    chi = get_chi(lin(r[-i:]), y[-i:], y_err[-i:], 2)
    
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
i_min = 30

# start plotting
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# plot the error in fit against number of measurements
ax1.plot(i_arr, err_arr[:, 0], ls='-', marker='.', color='royalblue', label='error $\sigma_c$')
ax1.axhline(y=err_arr[i_min, 0], ls='--', color='red', zorder=1)
ax1.axvline(x=i_min, ls='--', color='red', zorder=0)

# plot the point at which the error is the lowest
ax1.plot(i_min, err_arr[i_min, 0], ls='none', marker='x', color='red', label=f'i={i_min+1}')

ax2.plot(i_arr, chi_arr, ls='-', marker='.', color='green', label='reduced $\chi^2$')

# title and labels for plotting
plt.title(f'Fit Chracteristics against Number of Measurements')
ax1.set_xlabel('number of measurements $i$ ($unitless$)')
ax1.set_ylabel('error in $c$ from ECM $\sigma_c$ ($Bq \; m^2$)')
ax2.set_ylabel('reduced $\chi^2$ ($Bq \; m^2$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

fig.legend(bbox_to_anchor=(0.5,0.8))

plt.savefig(f'Plots/Distance/{dataset}_fit.png', dpi=300, bbox_inches='tight')
plt.show()

# perform curve fit to get line of best fit
opt, cov = so.curve_fit(fit_lin, r[-i_min:], y[-i_min:], sigma=y_err[-i_min:], absolute_sigma=True)

# convert coefficients from curve fit into function
lin = np.poly1d(opt)

# get the reduced chi squared value for the linear fit
chi = get_chi(lin(r[-i_min:]), y[-i_min:], y_err[-i_min:], 2)

# perform the linear fit using the upper and lower bounds on y
# opt_p, cov_p = np.polyfit(r[-(i_min):], y_p[-(i_min):], 1, cov=True)
# opt_m, cov_m = np.polyfit(r[-(i_min):], y_m[-(i_min):], 1, cov=True)
opt_p, cov_p = so.curve_fit(fit_lin, r[-i_min:], y_p[-i_min:], sigma=y_err[-i_min:], absolute_sigma=True)
opt_m, cov_m = so.curve_fit(fit_lin, r[-i_min:], y_m[-i_min:], sigma=y_err[-i_min:], absolute_sigma=True)

# find the error in the fit for the upper and lower bounds
err_p = np.sqrt(np.diagonal(cov_p))
err_m = np.sqrt(np.diagonal(cov_m))

# generate functions for fit from upper and lower bounds
# this time include systeamtic error as well as fit error
lin_p = np.poly1d(opt_p + err_p)
lin_m = np.poly1d(opt_m - err_m)

# calcualte source activity together with upper and lower bounds in Bq
A = lin(0) * 4 * np.pi / A_det
A_p = lin_p(0) * 4 * np.pi / A_det 
A_m = lin_m(0) * 4 * np.pi / A_det

# calcualte total errors in calcualted activity in Bq 
A_p_err = np.abs(A - A_p)
A_m_err = np.abs(A - A_m)

# load the theoretical values for an extended source generated from extended analysis
data_theoretical = np.loadtxt(f'Data/extended.csv', unpack=True, delimiter=',', skiprows=1)

# slice the data into distance rt in m, fraction ft, adjusted fraction yt
# ft and yt are both 2D arrys holding absolute value, upper and lower limits
rt = data_theoretical[0, :]
ft = data_theoretical[1:4, :]
yt = data_theoretical[4:8, :]

# cubic spline onto theoretical curve with air
cs_t = si.CubicSpline(rt, yt[0])
cs_t_p = si.CubicSpline(rt, yt[1])
cs_t_m = si.CubicSpline(rt, yt[2])

# function which returns scaled theoreticla curve for fitting
fit_t = lambda x, m, c : cs_t(x) * (m * x + c)
fit_t_p = lambda x, m, c : cs_t_p(x) * (m * x + c)
fit_t_m = lambda x, m, c : cs_t_m(x) * (m * x + c)

# use curve fit to get activity for theiretical curve in Bq and propagate total error
opt_t, cov_t = so.curve_fit(fit_t, r, y, sigma=y_err, absolute_sigma=True)
opt_t_p, cov_t_p = so.curve_fit(fit_t_p, r, y_p, sigma=y_err, absolute_sigma=True)
opt_t_m, cov_t_m = so.curve_fit(fit_t_m, r, y_m, sigma=y_err, absolute_sigma=True)

err_t_p = np.sqrt(np.diagonal(cov_t_p))
err_t_m = np.sqrt(np.diagonal(cov_t_m))

# scale the theoretical curve to match the data
yt_air = yt[0] * np.poly1d(opt_t)(rt)
yt_air_p = yt[1] * np.poly1d(opt_t_p + err_t_p)(rt)
yt_air_m = yt[2] * np.poly1d(opt_t_m - err_t_m)(rt)

# take another cubic spline for reduced chi squared calcualtion
cs_t = si.CubicSpline(rt, yt_air)

# calcualte reduced chi squared for the theoretical curve accounting for random errors
chi_t = get_chi(y, cs_t(r), y_err, 2)

# calcualte source activity from theoretical curve with upper and lower bounds due to total error
A_t = opt_t[1]
A_t_p = opt_t_p[1]
A_t_m = opt_t_m[1]

# calcualte total errors in calcualted activity in Bq 
A_t_p_err = np.abs(A_t - A_t_p)
A_t_m_err = np.abs(A_t - A_t_m)

# print numerical results
print()
print(f'dataset                         : {dataset}')
print(f'number of measurements          = {i_min+1}')
print()
print('results from linear fit:')
print(f'reduced chi squared             = {chi:.4g}')
print(f'calculated activity             = {A*1e-6:.4g} MBq')
print(f'upper bound activity            = {A_p*1e-6:.4g} MBq')
print(f'lower bound activity            = {A_m*1e-6:.4g} MBq')
print(f'activity with errors            = {A*1e-6:.5g} +{A_p_err*1e-6:.3g} -{A_m_err*1e-6:.3g} MBq')
print()
print(f'results from theoretical curve:')
print(f'reduced chi squared             = {chi_t:.4g}')
print(f'calculated activity             = {A_t*1e-6:.4g} MBq')
print(f'upper bound activity            = {A_t_p*1e-6:.4g} MBq')
print(f'lower bound activity            = {A_t_m*1e-6:.4g} MBq')
print(f'activity with errors            = {A_t*1e-6:.5g} +{A_t_p_err*1e-6:.3g} -{A_t_m_err*1e-6:.3g} MBq')


# array of offset distance for plotting in m
r_plot = np.linspace(0, np.amax(r), 1000)

# start plotting
fig, ax = plt.subplots(1, 1)

# plot theoretical curve for extended source
ax.plot(rt, yt_air, color='red', zorder=6, label='theoretical curve')
ax.fill_between(rt, yt_air_p, yt_air_m, color='red', zorder=4, alpha=0.25, label='total error')

# plot measured points
ax.plot(r, y, '.', markersize=2.5, color='black', zorder=7, label='measurement')
ax.plot(r[-i_min:], y[-i_min:], '.', markersize=2.5, color='blue', zorder=8, label='used in linear fit')

# plot errorbars for random error
ax.errorbar(r, y, xerr=r_err, yerr=y_err, fmt='none', color='teal', elinewidth=1.4, zorder=2, label='random error')

# plot region for systematic error
ax.fill_between(r, y_p, y_m, color='royalblue', alpha=0.4, zorder=1, label='systematic error', interpolate=True)

# plot line of best fit
ax.plot(r_plot, lin(r_plot), color='orange', zorder=5, label='linear fit')
ax.fill_between(r_plot, lin_p(r_plot), lin_m(r_plot), color='orange', alpha=0.3, zorder=3, label='total error')

# plot y intercept points
plt.plot([0, 0, 0], [lin(0), lin_p(0), lin_m(0)], '_', color='orange')

# plot vertical line at origin
plt.axvline(x=0, ls='-', linewidth=1, color='black')

# title and labels for plotting
plt.title('Adjusted Count Rate $\\frac{n}{\Delta t} r^2$ against Distance')
plt.xlabel('distance $r$ ($m$)')
plt.ylabel('adjusted count rate ($Bq \; m^2$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()
handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 3, 7, 4, 5, 6, 0, 1]
plt.legend([handles[i] for i in order], [labels[i] for i in order], markerscale=2)

plt.ylim(11, 19)

plt.savefig(f'Plots/Distance/{dataset}.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# plots for the lab report

# start plotting
fig, ax = plt.subplots(1, 1)

# plot theoretical curve for extended source
ax.plot(rt, yt_air, color='red', zorder=4, label='theoretical curve')
ax.fill_between(rt, yt_air_p, yt_air_m, color='red', zorder=3, alpha=0.25, label='total error in fit')

# plot measured points
ax.plot(r, y, '.', markersize=3, color='black', zorder=5, label='measurement')

# plot errorbars for random error
ax.errorbar(r, y, xerr=r_err, yerr=y_err, fmt='none', color='teal', elinewidth=1.4, zorder=2, label='random error')

# plot region for systematic error
ax.fill_between(r, y_p, y_m, color='royalblue', alpha=0.4, zorder=1, label='systematic error', interpolate=True)

# plot vertical line at origin
plt.axvline(x=0, ls='-', linewidth=1, color='black')

# title and labels for plotting
plt.title(r'Adjusted Count Rate $\frac{n}{\Delta t} r^2$ against Distance')
plt.xlabel('distance $r$ ($m$)')
plt.ylabel('adjusted count rate ($Bq \; m^2$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()
handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 4, 3, 0, 1]
plt.legend([handles[i] for i in order], [labels[i] for i in order], markerscale=2)

plt.ylim(11, 18.7)

plt.savefig('Plots/Distance/Fig_2.png', dpi=300, bbox_inches='tight')
plt.show()