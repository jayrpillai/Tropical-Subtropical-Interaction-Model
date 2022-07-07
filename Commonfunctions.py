# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 09:47:12 2022

@author: JayPillai
"""

import numpy as np
import xarray as xr
v = 5                         #m/s wind speed
g = 9.8                         #m/s^2 Accel due to gravity
ct = 1e-3                       #Dimensionless Transfer Coefficient
Rd = 287.058                    #J/(kg K) dry gas constant
div = 1e-5                      #Rate of large scale divergence in ST(for fixed max flux calculations)
                 
GammaD = 0.0098                 #K/m Dry Adiabatic Lapse Rate
L = 2.5e6                       #J/kg latent heat of vaporization for water
cp = 1005                       #J/(kg*K) heat capacity of air at constant pressure
Rv = 461.5                      #J/(kg*K) #moist gas constant




epsilon = 0.622                 #Dimensionless from Randall Buoyancy Flux Paper
delta = 0.608                   #Dimensionless from Randall Buoyancy Flux Paper
eps2 = 0.1                      #Dimensionless from Randall Buoyancy Flux Paper

#Terms for Slab Ocean(s)
c = 4184                        #J/(kg*K) Specific heat of ocean water
D = 10                         #meters   Depth of Subtropical ocean
rhowater = 1000                 #kg/m^3   Density of Ocean Water
DTrop = 10                      #meters   Depth of Tropical Ocean

#Entrainment Calculation Terms
cd = 1e-3                       #Drag coefficient
ce0 = 1e-3                      #Entrainment coefficient from new Parameterization
ce1 = 1e-2                      #Entrainment coefficient from new Parameterization


#Radiation Terms
albmax = 0.5                    #Max albedo for Strato Cloud
k = 40                          #m^2/kg coefficient in exponent
albss = 0.06                    #Albedo of Sea Surface
S_TOA_Dn_ST = 471               #W/m^2 Incoming Solar Radiation in Subtropics
S_TOA_Dn_TR = 390               #W/m^2 Incoming Solar Radiation in Tropics (ST is greater than TR because we assume Summer season)
sb = 5.67e-8                    #W/(m^2 K^4) Stefan-Boltzmann Constant


anvilmax = 0.8                  #Max albedo of Anvil cloud in Tropics
k2 = 50                         #m^2/kg constant for Ice Water Path calculation in anvil albedo
tauice = 1e3                    #time constant for Ice Water Path production

#Tropical Constants
P0 = 7.97e-10                   #mm/second, default Precip rate for exponential relation
alphad = 14.72                  #Dimensionless Coefficient from Rushley et al. 2018, equation 4 version 7
                                #from Rushley et al. 2018, equation 4 version 7

acrea = 0.18664                 #W/m^2
acreb = 0.06923                 #inverse of CRH, no units
lwacrea = 0.0392974             #W/m^2
lwacreb = 0.0833349169          #inverse of CRH
#Precipitable water constants
e0 = 100                        #100 Pa = 1 mb From Kelley et al.
Ae = 21.656                     # From Kelley et al.
Be = 5418                       #Kelvin From Kelley et al.
T0 = 273                        #Kelvin From Kelley et al.

#LW And SW CRE AT TOA AND SFC Coefs for CRE = alpha * (np.exp(Beta * crh**2) - 1)
CRHcoeff = np.log(2)
alphaLWSFC = 68.26558891857995
alphaLWTOA = 224.5182069370315
alphaSWSFC = -348.5258389335704
alphaSWTOA = -322.0043497633839

#Transmittance Coeffs
mu = 0.2
nu = 0.03
J_0 = 0.95
#f3 coefs
af3 = 0.9
bf3 = 0.11
#f1 Coefs
cf1 = 0.73
df1 = 0.003



#Clausius Clapeyron Relation
def ClausClap(T):
    es= 611 * np.exp((L/Rv)*(1/273 -1/T))
    return es #units of Pascals

#Water vapor mixing ratio function
def qs(T,p):
    qs = epsilon*ClausClap(T)/p
    return qs #no units since it is a ratio

#variable air density assuming ideal gas law to calculate air density at any pressure and temperature
def rho(T,p):
    rho = p/(Rd*T)
    return rho

def rho_layer(z , rho_top, Gamma, TempTop, zt):
    rho_layer = rho_top * (1 + (Gamma/TempTop) * (zt - z)) ** (( g / (Gamma * Rv)) - 1)
    return rho_layer

#Define Moist Adiabatic Lapse Rate
def MoistGamma(T,p):
    GammaMS = g*(1 + L * qs(T,p) / (Rd * T)) / (cp + (qs(T,p) * L**2)/( Rv * T**2))
    return GammaMS

def F(T): #Function to calculate the F in KElly et al
    F = epsilon * e0 * T0 * np.exp(Ae - Be/T)/(Rd*Be)
    return F


def Integral_zbplus_1(z_n,num_eval_points, Gamma, TempTop, zt, zb_n, Sigma_ST_top):
    zbplus = 1 + (Gamma / TempTop) * (zt - zb_n)
    Integral_zbplus_1 =  ((zbplus - 1) / num_eval_points) * (TempTop / Gamma) * \
        Simpson(z_n, num_eval_points, Sigma_ST_top, Gamma)
    return Integral_zbplus_1

def Simpson(z_n, num_eval_points, Sigma_ST_top, Gamma):
    Sum = 0
    Func = 1 / (1 - Sigma_ST_top * z_n **(1 - g / (Gamma * Rv)))
    for i in range(0,num_eval_points):
        if i == 0 or i == 100:
            c_n = 1
        elif i % 2 == 0:
            c_n = 2
        else:
            c_n = 4
    
        Sum += c_n * Func[i]
        
    Simpson = Sum
    return Simpson



def f1TR(W,CO2level): #Ratio of Rup sfc clr to OLR clr in Tropics
    #RupTR = cf1 - df1 * W
    epsbulk = 0.6
    return epsbulk


def f1ST(W,CO2level): #ratio of Rup Sfc clr to OLR clr in Subtropics
    #RupST = cf1 - df1 * W
    epsbulk = 0.7
    return epsbulk


def Trans_TR(W,CO2level): #Transmittance of atmosphere in Tropics
    #J = J_0 - mu * (1 - np.exp( - nu * W))
    J = 0.85
    return J


def Trans_ST(W,CO2level): #Tranmittance of atmosphere in Subtropics
    #J = J_0 - mu * (1 - np.exp( - nu * W))
    J = 0.95
    return J


def f3ST(W,CO2level):
    #f3 = af3 * (1 - np.exp(-bf3 * W))
    f3 = 0.7
    return f3


def f3TR(W,CO2level):
    #f3TR = af3 * (1 - np.exp(-bf3 * W))
    f3 = 0.85
    return f3

def CREFactor(CRH):
    postfactor = np.exp(CRHcoeff * CRH**2) - 1
    return postfactor

def create_variable_dataarray(namelist_var, dstore, coords, dims):
    da = xr.DataArray(
        data   = dstore[namelist_var], # <- pull the data directly from the dstore dict
        dims   = dims, # <- this should be in the same order as the np array "q"
        coords = coords ,
        #attrs  = metadata_var[namelist_var] # Pull metadata from dictionary
    )

    return da

def save_to_disk(namelist_vars, dstore, save_metadata, coords,dims):
    # Create an xarray dataset object to store the data, will add data a bit later
    save_ds = xr.Dataset(attrs = save_metadata)
    
    #Iterate over each variable in the namelist and include in the dataset
    for namelist_var in namelist_vars:
        da = create_variable_dataarray(namelist_var, dstore, coords, dims)
        
        save_ds[namelist_var] = da
        print(namelist_var)


