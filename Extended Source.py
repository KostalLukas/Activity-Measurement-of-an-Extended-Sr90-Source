# -*- coding: utf-8 -*-
"""
Radioactivity Extended Source v1.1

Lukas Kostal, 9.3.2023, ICL
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.special as ss


# define the function to be integrated using numerical integration
def func(x, d, r_s, r_d):
    y = np.exp(-d * x) * ss.j1(r_s * x) * ss.j1(r_d * x) / x
    
    return y

# define the large d approximation for the solid angle for comparison
def approx(d, r_s, r_d):
    
    d[0] += 1e-6
    
    a = (r_d / d)**2
    b = (r_s / d)**2
    
    y = 1
    
    y += - 1/(1+b)**(1/2) - 3/8 * a*b / (1 + b)**(5/2)
    y += a**2 * (5/16 * b / (1+b)**(7/2) - 35/16 * b**2 / (1+b)**(9/2))
    y += a**3 * (35/128 * b / (1+b)**(9/2) - 315/256 * b**2 / (1+b)**(11/2) + 1155/1024 * b**3 / (1+b)**(13/2))
    
    y *= 2*np.pi
    
    return y 


# specify detector area in m^2
A_d = 1e-4

# specify source radius in m
r_s = 9.17e-3
r_err = 0.02e-3

# calcualte radius of equaivalent circular detector
r_d = np.sqrt(A_d / np.pi)

d_min = 0
d_max = 1
d_n = 500

# array of distances between center of source and detector
d_arr = np.linspace(0, 1, d_n)

# array to hold values of solid angles at the distances
Omg_arr = np.zeros(d_n)
Omg_arr_p = np.zeros(d_n)
Omg_arr_m = np.zeros(d_n)

# limit on the area of trapezium at which to truncate the numerical integration
A_lim = 1e-8

# limit on the value of x at which to truncate the numerical integration
x_lim = 100

# incrmeent in x so width of trapezium in numerical integration
x_inc = 1

# loop over all values for distance between source and detector
for i in range(0, len(d_arr)):
    
    # current distance between source and detector
    d = d_arr[i]
    
    # start at the first incrmeent in x since 0 would diverge
    x = A_lim
    
    # area of one trapezium element to be added to overall integral value
    A = 0
    A_p = 0
    A_m = 0
    
    # value of the integral to be updated and associated error
    A_sum = 0
    A_sum_p = 0
    A_sum_m = 0
    
    # perform numerical integration untill both limiting conditions are met
    while (A > A_lim or x < x_lim):
        # calcualte area of trapezium element
        A = x_inc * (func(x + x_inc, d, r_s, r_d) + func(x, d, r_s, r_d)) / 2
        A_p = x_inc * (func(x + x_inc, d, r_s + r_err, r_d) + func(x, d, r_s + r_err, r_d)) / 2
        A_m = x_inc * (func(x + x_inc, d, r_s - r_err, r_d) + func(x, d, r_s - r_err, r_d)) / 2
        
        # add are of trapezium element to integral value
        A_sum += A 
        A_sum_p += A_p
        A_sum_m += A_m
        
        # increment the value of x
        x += x_inc
    
    # calcualte solid angle from the value of the integral
    Omg_arr[i] = 4 * np.pi * r_d / r_s * A_sum
    Omg_arr_p[i] = 4 * np.pi * r_d / (r_s + r_err) * A_sum
    Omg_arr_m[i] = 4 * np.pi * r_d / (r_s - r_err) * A_sum

# array of values for solid angle using large d approximation formula
Omg_approx = approx(d_arr, r_s, r_d)
Omg_approx_p = approx(d_arr, r_s + r_err, r_d)
Omg_approx_m = approx(d_arr, r_s - r_err, r_d)

# fraction of emitted radiation quanta detected by detector from numerical integration
g = Omg_arr / (4 * np.pi)
g_p = Omg_arr_p / (4 * np.pi)
g_m = Omg_arr_m / (4 * np.pi)

# fraction of emitted radiation quanta detected by detector from approximation
g_approx = Omg_approx / (4 * np.pi)
g_approx_p = Omg_approx_p / (4 * np.pi)
g_approx_m = Omg_approx_m / (4 * np.pi)

# adjusted fraction of emitted radiation quanta detected by detector
y = g * d_arr**2
y_p = g_p * d_arr**2
y_m = g_m * d_arr**2

y_approx = g_approx * d_arr**2
y_approx_p = g_approx_p * d_arr**2
y_approx_m = g_approx_m * d_arr**2

# join the results from numerical integration only to be written into a .csv file
data_out = np.stack((d_arr, g, g_p, g_m, y, y_p, y_m), axis=-1)

# write the parameters for the simulation into a .csv file
hdr = 'distance (m), g, g_p, g_m, y (m^2), y_p (m^2), y_m (m^2)'
np.savetxt('Output/extended.csv', data_out, header=hdr ,delimiter=',', newline='\n')

# plot fraction of detected quanta against distance
plt.plot(d_arr, g, color='blue', label='numerical integration')
plt.plot(d_arr, g_approx, color='red', label='Taylor series')

plt.fill_between(d_arr, g_p, g_m, color='blue', alpha=0.4)
plt.fill_between(d_arr, g_approx_p, g_approx_m, color='red', alpha=0.4)

# title and labels for plotting
plt.title('Fraction of Detected Quanta against Distance')
plt.xlabel('distance $r$ ($m$)')
plt.ylabel('fraction of detected quanta $g$ ($unitless$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()
plt.legend()

# save the plot
plt.savefig('Plots/Extended/fraction.png', dpi=300, bbox_inches='tight')
plt.show()

# plot adjusted fraction of detected quanta against distance
plt.plot(d_arr, y, color='blue', label='numerical integration')
plt.plot(d_arr, y_approx, color='red', label='Taylor series')

plt.fill_between(d_arr, y_p, y_m, color='blue', alpha=0.4)
plt.fill_between(d_arr, y_approx_p, y_approx_m, color='red', alpha=0.4)

# title and labels for plotting
plt.title('Adjusted Fraction of Detected Quanta against Distance')
plt.xlabel('distance $r$ ($m$)')
plt.ylabel('adjusted fraction $g \: r^2$ ($m^2$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()
plt.legend()

# save the plot
plt.savefig('Plots/Extended/fraction_adjusted.png', dpi=300, bbox_inches='tight')
plt.show()
