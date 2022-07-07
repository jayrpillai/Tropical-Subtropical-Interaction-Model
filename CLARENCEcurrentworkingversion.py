# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 18:24:35 2022

@author: JayPillai
"""




import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime as dt
from Commonfunctions import *


import xarray as xr

logging.basicConfig(filename = "Errors1.log", level = logging.DEBUG, filemode = 'w')
###################################################################################################################
"""
This is CLARENCE, the numerical model of the tropcial and subtropical atmospheres created 
by Jaydeep Pillai and David Randall at CSU, 2021. 

We finished preliminary tests in August but an injury gave us a small setback. 
We restarted the project on September 1st, 2021.
We presented at AGU 2021, and started back up in late January to finish the model.
"""


"""Declare Variables """


Model_name = "CLARENCE"
Model_version = "V2"
Model_case = "NewSigma"
Creator = "Jay"
currenttime = dt.datetime.now()
Start_time = 1#currenttime.strftime("%H%M_%d_%m_%Y")








v = 5                           #m/s wind speed
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
OzoneA = 0.04
#f3 coefs
af3 = 0.9
bf3 = 0.11
#f1 Coefs
cf1 = 0.73
df1 = 0.003

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
        #print(namelist_var)
        
    
   
   # # print("saving now")
    filestring = f'\{Model_name}{Model_version}{Model_case}_{Start_time}'
    path = r'C:\Users\JayPillai\Desktop\CLARENCEOUTPUT'
    save_ds.to_netcdf(path + filestring) # <- Save to disk
    #print("finished")


save_metadata = {
    "MODEL_NAME": Model_name,
    "MODEL_VERSION": Model_version,
    "MODEL_CASE": Model_case,
    "created_by": Creator,        
    "START_TS": Start_time, # <- Just pull the current time
}





#namelist of model outputs, follows the order that arrays are defined below as well!


namelist_vars = ["h_m", "ql_m", "deltap_m", "SST_ST", "SST_TR", "z_B",
 
                 "p_B", "z_C", "LWP", 

		 "T_c", "deltas_v_minus_deltas_v_crit", "B_Flux", "Ent", "W_E", "G0", "G1",

		 "q_B", "l_B", "T_B", "delta(q+l)", "deltah", "h_B+", "T_B+",
		 "AVG BL Rho",

		 "Fh_ss", "Fq_ss", "Fh_B", "Fq_B", "Fh_E", "Fh_R", "Fh_Solar", 
		 "Fq_E", "Fh_M", "Fq_M", "Shear",

		 "DeltaR", "R_SFC_Net_ST", "R_BL_Net_ST", "R_BL+_Net_ST", 
		 "R_TOA_Net_ST", "R_SFC_Net_CLR_ST", 

		 "R_SFC_Up_ST", "R_TOA_CLR_ST", "R_BL_Up_CLR_ST", "R_BL+_Up_CLR_ST",
		 "R_BL+_Up_ST", "R_BL_Up_ST", 

		 "R_SFC_Dn_CLR_ST", "R_SFC_Dn_ST", "R_BL+_Dn_ST", "R_BL_Dn_ST",

		 "W_FT", "W_B", "W_ST", "J_FT", "J_BL", "J_ST",

		 "S_TOA_Dn_CLR_ST", "S_BL+_Dn_CLR_ST", "S_SFC_Dn_CLR_ST",

		 "S_SFC_Up_CLR_ST", "S_BL+_Up_CLR_ST", "S_TOA_Up_CLR_ST", 

		 "S_SFC_Net_CLR_ST", "S_BL+_Net_CLR_ST", "S_TOA_Net_CLR_ST", 
		 "S_Inf_Net_CLR_ST", 

		 "S_BL+_Dn_ST", "S_SFC_Dn_ST", "S_TOA_Dn_ST", 

		 "S_SFC_Up_ST", "S_BL+_Up_ST", "S_TOA_Up_ST",  

		 "S_SFC_Net_ST", "S_BL+_Net_ST", "S_TOA_Net_STs", "S_Flux_Diff_FT_ST",
	
		 "Evap", "Precip", "CRH", "Z_T", "T_T", "Gamma_0", "P_T", "W_Max", 
		 "W_TR", "q_T", "q_up", "q_S_TR", "h_S_TR", 
		 
		 "J_TR", "SW_CRE_SFC", "SW_CRE_TOA", "ACRE_SW",

		 "S_SFC_Dn_CLR_TR", "S_SFC_Up_CLR_TR", "S_SFC_Net_CLR_TR", 
		 "S_TOA_Up_CLR_TR", "S_TOA_Net_CLR_TR",
	
		 "S_SFC_Net_TR", "S_TOA_Net_TR",
		
		 "R_SFC_Up_TR", "R_SFC_Dn_CLR_TR", "R_TOA_CLR_TR",

		 "LW_CRE_SFC", "LW_CRE_TOA", "ACRE_LW", "LW_ALLSKY_FD_TR", "LW_CLRSKY_FD_TR",

		 "R_TOA_Net_TR", "R_SFC_Net_TR",

		 "Int_Q_TR_dz", "Q_TR", "Int_Q_ST_dz", "Q_ST", 

		 "Sigma_TR", "Sigma_ST", "Mass Flux", "q_Dn"]







#Create containers to store the data
#Subtropical Lists





logging.debug(f'Function started at {dt.datetime.now()}')


""" 
Main Function
"""

#main function
def Main( days, h_i, q_i, deltapm_i, SST_ST_i, p_n, SST_TR_i, CO2level ):
    
    #days = 50
    timestep_length = 100               #seconds time step
    tstep = timestep_length
    secondsinday = 86400                #Seconds in one day
    steps_per_collection = 5            #Timesteps per data collection into arrays

    total_steps = int(days * secondsinday / timestep_length)
    total_length = int(days * secondsinday)
    timescale = np.linspace(0, total_length, total_steps )    #Time the simulation will run for
    times = np.zeros(total_steps // steps_per_collection) 

    hms = np.zeros_like(times)       ; hms[:] = np.nan        #Moist Static Energy
    qms = np.zeros_like(times)       ; qms[:] = np.nan     #Water Mixing Ratio
    deltapms = np.zeros_like(times)  ; deltapms[:] = np.nan
    SSTs = np.zeros_like(times)      ; SSTs[:] = np.nan     #Sea Surface Temperatures Subtropics
    zbs = np.zeros_like(times)       ; zbs[:] = np.nan   #Boundary Layer Depth

    pbs = np.zeros_like(times)       ; pbs[:] = np.nan     #Boundary Layer Pressure
    zcs = np.zeros_like(times)       ; zcs[:] = np.nan     #Cloud Base Height
    LWPs = np.zeros_like(times)      ; LWPs[:] = np.nan    #Liquid Water Path

    Tcs = np.zeros_like(times)       ; Tcs[:] = np.nan     #Cloud Base Temperature
    svs = np.zeros_like(times)       ;svs[:] = np.nan     #Virtual Dry Static Energy
    B_Es = np.zeros_like(times)      ;B_Es[:] = np.nan       #Buoyancy Flux Integral of Boundary Layer
    Ents = np.zeros_like(times)      ;Ents[:] = np.nan     #Entrainment Flux Rate
    W_Es = np.zeros_like(times)      ;W_Es[:] = np.nan     #Entrainment Velocity
    G0s = np.zeros_like(times)       ;G0s[:] = np.nan     #Constant in Dave's Equation
    G1s = np.zeros_like(times)       ;G1s[:] = np.nan     #Coefficient in Dave's Equation


    qbs = np.zeros_like(times)       ;qbs[:] = np.nan     #Boundary Layer Water Mixing Ratio
    lbs = np.zeros_like(times)       ;lbs[:] = np.nan     #Boundary Layer Liquid Water Mixing Ratio
    Tbs = np.zeros_like(times)       ;Tbs[:] = np.nan     #Boundary Layer Temperature
    deltaqls = np.zeros_like(times)  ;deltaqls[:] = np.nan     #Change in Water Mixing Ratio across Boundary Layer
    deltahs = np.zeros_like(times)   ;deltahs[:] = np.nan      #Change in Moist Static Energy across Boundary Layer
    hbpluses = np.zeros_like(times)  ;hbpluses[:] = np.nan     #Just Above Boundary Layer Moist Static Energy
    Tbpluses = np.zeros_like(times)  ;Tbpluses[:] = np.nan     #Just Above Boundary Layer Temperature
    qbpluses = np.zeros_like(times)  ;qbpluses[:] = np.nan     #Just above Boundary Layer Moisture content
    sbpluses = np.zeros_like(times)  ;sbpluses[:] = np.nan     #Just above boundary layer dry static energy
    avgrhos = np.zeros_like(times)   ;avgrhos[:] = np.nan      #Average Density of the Boundary Layer


      

    Fhss = np.zeros_like(times)      ;Fhss[:] = np.nan     #Surface Moist Static Energy Flux
    Fqss = np.zeros_like(times)      ;Fqss[:] = np.nan     #Surface Water Mixing Ratio Flux
    Fhbs = np.zeros_like(times)      ;Fhbs[:] = np.nan     #Boundary Layer Moist Static Energy Flux
    Fqbs = np.zeros_like(times)      ;Fqbs[:] = np.nan     #Boundary Layer Water Mixing Ratio Flux
    Fhes = np.zeros_like(times)      ;Fhes[:] = np.nan     #Energy from entrained air across the boundary layer.
    Fhrs = np.zeros_like(times)      ;Fhrs[:] = np.nan     #Energy radiating from surface to BDL Top
    Fhsolars = np.zeros_like(times)  ;Fhsolars[:] = np.nan #Energy radiating from Sun, from BDL Top to Surface
    Fqes = np.zeros_like(times)      ;Fqes[:] = np.nan     #Moisture entrained into BDL
    FhMs = np.zeros_like(times)      ;FhMs[:] = np.nan     #Energy brought to BDL by Mass Flux
    FqMs = np.zeros_like(times)      ;FqMs[:] = np.nan     #Moisture brought to BDL by Mass Flux
    Shears = np.zeros_like(times)    ;Shears[:] = np.nan   #Energy from Shear in the BDL


    #Subtropical Radiation Lists
    #Net LW ST Radiation
    DeltaRs = np.zeros_like(times)            ;DeltaRs[:] = np.nan         #Change in LW Radiation over the Boundary Layer
    R_SFC_Net_STs = np.zeros_like(times)      ;R_SFC_Net_STs[:] = np.nan   #Surface LW Net Radiation in Subtropics
    R_BL_Net_STs = np.zeros_like(times)       ;R_BL_Net_STs[:] = np.nan    #Boundary Layer LW Net Radiation Subtropics
    R_BLplus_Net_STs = np.zeros_like(times)   ;R_BLplus_Net_STs[:] = np.nan#Just Above Boundary Layer LW Net Radiation Subtropics
    R_TOA_Net_STs = np.zeros_like(times)      ;R_TOA_Net_STs[:] = np.nan   #TOA LW Net Radiation in Subtropics
    R_SFC_Net_CLR_STs = np.zeros_like(times)  ;R_SFC_Net_CLR_STs[:] = np.nan #SFC LW Net Clrsky Radiation Subtropics

    #Upward LW Radiation
    R_SFC_Up_STs = np.zeros_like(times)       ;R_SFC_Up_STs[:] = np.nan    #SFC LW Upwards Subtropics
    R_TOA_CLR_STs = np.zeros_like(times)      ;R_TOA_CLR_STs[:] = np.nan   #TOA Clrsky LW Upwards Subtropics
    R_BL_Up_CLR_STs = np.zeros_like(times)    ;R_BL_Up_CLR_STs[:] = np.nan #BDL Clrsky LW Upwards Subtropics
    R_BLplus_Up_CLR_STs = np.zeros_like(times);R_BLplus_Up_CLR_STs[:] = np.nan #Just Above BDL LW Upward Subtropics
    R_BLplus_Up_STs = np.zeros_like(times)    ;R_BLplus_Up_STs[:] = np.nan #Just above Boundary Layer LW Upwards Subtropics
    R_BL_Up_STs = np.zeros_like(times)        ;R_BL_Up_STs[:] = np.nan     #BDL LW Upwards Subtropics

    #Downward LW Radiation
    R_SFC_Dn_CLR_STs = np.zeros_like(times)  ;R_SFC_Dn_CLR_STs[:] = np.nan    #SFC downward CLRsky LW Subtropics
    R_SFC_Dn_STs = np.zeros_like(times)      ;R_SFC_Dn_STs[:] = np.nan     #SFC Downward LW Subtropics
    R_BLplus_Dn_STs  = np.zeros_like(times)  ;R_BLplus_Dn_STs[:] = np.nan  #Just above BDL Downward LW Subtropics
    R_BL_Dn_STs = np.zeros_like(times)       ;R_BL_Dn_STs[:] = np.nan      #BDL Downwards LW Subtropics   



    #Water vapor in each layer and occuring transmissivity of layer
    W_FTs = np.zeros_like(times)   ;W_FTs[:] = np.nan    #Water Vapor in Free Troposphere
    W_Bs = np.zeros_like(times)    ;W_Bs[:] = np.nan     #Water Vapor in Boundary Layer
    W_STs = np.zeros_like(times)   ;W_STs[:] = np.nan    #Water Vapor in Total Subtropical column
    J_FTs = np.zeros_like(times)   ;J_FTs[:] = np.nan    #Transmittance of Free Troposphere
    J_BLs = np.zeros_like(times)   ;J_BLs[:] = np.nan    #Transmittance of Boundary Layer
    J_STs = np.zeros_like(times)   ;J_STs[:] = np.nan    #Transmittance of Total Subtropical Column

    #Downward clear sky solar
    S_TOA_Dn_CLR_STs = np.zeros_like(times)         ;S_TOA_Dn_CLR_STs[:] = np.nan   #TOA Downward Clrsky SW Subtropics
    S_BLplus_Dn_CLR_STs = np.zeros_like(times)      ;S_BLplus_Dn_CLR_STs[:] = np.nan#Just above BDL Downward Clrsky SW Subtropics
    S_SFC_Dn_CLR_STs = np.zeros_like(times)         ;S_SFC_Dn_CLR_STs[:] = np.nan   #SFC Downward Clrsky SW Subtropics

    #Upward clear sky solar
    S_SFC_Up_CLR_STs = np.zeros_like(times)         ;S_SFC_Up_CLR_STs[:] = np.nan   #SFC Upward Clrsky SW Subtropics
    S_BLplus_Up_CLR_STs = np.zeros_like(times)      ;S_BLplus_Up_CLR_STs[:] = np.nan#Just above BDL Upward Clrsky Subtropics
    S_TOA_Up_CLR_STs = np.zeros_like(times)         ;S_TOA_Up_CLR_STs[:] = np.nan   #TOA Upwards SW ClrSky Subtropics

    #Net clear sky solar
    S_SFC_Net_CLR_STs = np.zeros_like(times)        ;S_SFC_Net_CLR_STs[:] = np.nan  #SFC Net Clrsky SW Subtropics
    S_BLplus_Net_CLR_STs = np.zeros_like(times)     ;S_BLplus_Net_CLR_STs[:] = np.nan #Just above BDL Net Clrsky SW Subtropics
    S_TOA_Net_CLR_STs = np.zeros_like(times)        ;S_TOA_Net_CLR_STs[:] = np.nan  #TOA Net Clrsky SW Subtropics
    S_Inf_Net_CLR_STs = np.zeros_like(times)        ;S_Inf_Net_CLR_STs[:] = np.nan  #ClrSky SW at infinity(outside of atmosphere) Subtropics    

    #Downward full sky solar
    S_BLplus_Dn_STs = np.zeros_like(times)    ;S_BLplus_Dn_STs[:] = np.nan   #Just above BDL Downward SW Subtroics
    S_SFC_Dn_STs = np.zeros_like(times)       ;S_SFC_Dn_STs[:] = np.nan      #SFC Downward SW Subtropics
    S_TOA_Dn_STs = np.zeros_like(times)       ;S_TOA_Dn_STs[:] = np.nan      #Downward permeating SW radiation Subtropics(Constant at 471 W/m^2)

    #Upward full sky solar
    S_SFC_Up_STs = np.zeros_like(times)       ;S_SFC_Up_STs[:] = np.nan      #SFC Upward SW Subtropics
    S_BLplus_Up_STs =np.zeros_like(times)     ;S_BLplus_Up_STs[:] = np.nan   #Just above BDL Upward SW Subtropics
    S_TOA_Up_STs = np.zeros_like(times)       ;S_TOA_Up_STs[:] = np.nan      #TOA Upward SW Subtropics

    #Net full sky solar
    S_SFC_Net_STs = np.zeros_like(times)      ;S_SFC_Net_STs[:] = np.nan     #SFC Net SW Subtropics
    S_BLplus_Net_STs = np.zeros_like(times)   ;S_BLplus_Net_STs[:] = np.nan  #Just above BDL Net SW Subtropics
    S_TOA_Net_STs =np.zeros_like(times)       ;S_TOA_Net_STs[:] = np.nan     #TOA Net SW Subtropics
    S_Flux_Diff_FT_STs = np.zeros_like(times) ;S_Flux_Diff_FT_STs[:] = np.nan#Flux difference across Free Troposphere SW Subtropics


    #Tropical Lists
    Es = np.zeros_like(times)                 ;Es[:] = np.nan       #Evaporation Rates
    Ps = np.zeros_like(times)                 ;Ps[:] = np.nan       #Precipitation Rates
    CRHs = np.zeros_like(times)               ;CRHs[:] = np.nan     #Columns Relative Humidity
    SST_TRs = np.zeros_like(times)            ;SST_TRs[:] = np.nan  #Tropical SST          
    zts = np.zeros_like(times)                ;zts[:] = np.nan      #Cloud Top Height in Tropics
    TempTops = np.zeros_like(times)           ;TempTops[:] = np.nan #Temp at Cloud Top Tropics
    Gamma0s = np.zeros_like(times)            ;Gamma0s[:] = np.nan  #Env. Lapse rate of Tropics
    pts = np.zeros_like(times)                ;pts[:] = np.nan      #Pressure at Cloud Top
    Wmaxes = np.zeros_like(times)             ;Wmaxes[:] = np.nan   #Max Water Vapor in Tropical Column
    W_TRs = np.zeros_like(times)              ;W_TRs[:] = np.nan    #Amount of Water Vapor in Tropical Column
    q_tops = np.zeros_like(times)             ;q_tops[:] = np.nan   #q_sat at Cloud Top
    q_ups = np.zeros_like(times)              ;q_ups[:] = np.nan    #Water Vapor being sent to Subtropics
    q_S_TRs = np.zeros_like(times)             ;q_S_TRs[:] = np.nan   #Water Vapor at surface Tropics
    h_S_TRs = np.zeros_like(times)             ;h_S_TRs[:] = np.nan   #MSE at surface Tropics

    #Tropical Radiation lists

    J_TRs = np.zeros_like(times)                ;J_TRs[:] = np.nan         #Transmittance of Tropical Column
        #Cloud effects
    SW_CRE_SFCs = np.zeros_like(times)          ;SW_CRE_SFCs[:] = np.nan   #SW Cloud Radiative Effect at Surface
    SW_CRE_TOAs = np.zeros_like(times)          ;SW_CRE_TOAs[:] = np.nan   #SW Cloud Radiative Effect at TOA     
    ACRE_SWs = np.zeros_like(times)             ;ACRE_SWs[:] = np.nan      #Atmospheric Cloud Radiative Effect in SW(TOA-SFC)

        #Clr Sky
    S_SFC_Dn_CLR_TRs = np.zeros_like(times)     ;S_SFC_Dn_CLR_TRs[:] = np.nan      #SW SFC Downward ClrSky Tropics
    S_SFC_Up_CLR_TRs = np.zeros_like(times)     ;S_SFC_Up_CLR_TRs[:] = np.nan      #SW SFC Upward ClrSky Tropics
    S_SFC_Net_CLR_TRs = np.zeros_like(times)    ;S_SFC_Net_CLR_TRs[:] = np.nan     #SW SFC Net ClrSky Tropics
    S_TOA_Up_CLR_TRs = np.zeros_like(times)     ;S_TOA_Up_CLR_TRs[:] = np.nan      #SW TOA Upwards ClrSky Tropics
    S_TOA_Net_CLR_TRs = np.zeros_like(times)    ;S_TOA_Net_CLR_TRs[:] = np.nan     #SW TOA Net ClrSky Tropics

        #Net Radiation at Top and Bottom
    S_SFC_Net_TRs = np.zeros_like(times)        ;S_SFC_Net_TRs[:] = np.nan         #SW SFC Net Tropics
    S_TOA_Net_TRs = np.zeros_like(times)        ;S_TOA_Net_TRs[:] = np.nan         #SW TOA Net Tropics


        #LW Radiation 
    R_SFC_Up_TRs = np.zeros_like(times)         ;R_SFC_Up_TRs[:] = np.nan          #LW SFC Upwards Tropics
    R_SFC_Dn_CLR_TRs = np.zeros_like(times)     ;R_SFC_Dn_CLR_TRs[:] = np.nan      #LW SFC Downwards ClrSky Tropics
    R_TOA_CLR_TRs = np.zeros_like(times)        ;R_TOA_CLR_TRs[:] = np.nan         #LW TOA ClrSky Tropics



    LW_CRE_SFCs = np.zeros_like(times)              ;LW_CRE_SFCs[:] = np.nan                #LW Cloud Radiative Effect Surface
    LW_CRE_TOAs = np.zeros_like(times)              ;LW_CRE_TOAs[:] = np.nan                #LW Cloud Radiative Effect TOA
    ACRE_LWs = np.zeros_like(times)                 ;ACRE_LWs[:] = np.nan                   #LW Atmospheric Cloud Radiative Effect 
    LW_ALLSKY_FLUX_DIFF_TRs = np.zeros_like(times)  ;LW_ALLSKY_FLUX_DIFF_TRs[:] = np.nan    #All Sky Flux Difference across Tropical Column LW
    LW_CLRSKY_FLUX_DIFF_TRs = np.zeros_like(times)  ;LW_CLRSKY_FLUX_DIFF_TRs[:] = np.nan    #Clr Sky Flux Difference across Tropical Column LW

    R_TOA_Net_TRs = np.zeros_like(times)            ;R_TOA_Net_TRs[:] = np.nan              #LW Net TOA Tropics
    R_SFC_Net_TRs = np.zeros_like(times)            ;R_SFC_Net_TRs[:] = np.nan              #LW Net SFC Tropics



    #Coupling Lists

    Int_Q_TR_dzs = np.zeros_like(times)     ;Int_Q_TR_dzs[:] = np.nan   #Integrate Radiative Heating of TR
    Q_TRs = np.zeros_like(times)            ;Q_TRs[:] = np.nan          #Heating Rate of Tropics
    Int_Q_ST_dzs = np.zeros_like(times)     ;Int_Q_ST_dzs[:] = np.nan   #Integrated Radiative cooling of ST
    Q_STs = np.zeros_like(times)            ;Q_STs[:] = np.nan          #Cooling Rate of ST

    Sigma_TRs = np.zeros_like(times)        ;Sigma_TRs[:] = np.nan      #Tropical Box Fraction
    Sigma_STs = np.zeros_like(times)        ;Sigma_STs[:] = np.nan      #Subtropical Box Fraction
    Ms = np.zeros_like(times)               ;Ms[:] = np.nan             #Mass Flux
    q_dns = np.zeros_like(times)            ;q_dns[:] = np.nan          #Moisture flowing from ST to TR


    
    #Initial variables
    h_n = h_i               #Initial Moist Static Energy
    q_n = q_i               #Initial Water Mixing Ratio
    deltapm_n = deltapm_i   #Initial Boundary Layer Depth
    SST_ST_n = SST_ST_i     #Initial Subtropical SST
    SST_TR_n = SST_TR_i     #Initial Tropical SST

    
    
   
    
    for timestep in range(0, total_steps ):
        
        
        #Tropical system solution
        if timestep == 0:
           M = 0.02
           CRH = 0.7
           q_top = 0
           Sigma_TR_SFC = 0.85
           
           E = rho(SST_TR_n, p_n) * v * ct * qs(SST_TR_n, p_n) * (1-CRH)
                                                                      #Evaporation rate of tropics
           
           P = E + M * (q_n - q_top) / Sigma_TR_SFC                      #Moisture budget of tropics, solved for precipitation rate
       
           CRH = np.log( P / P0) / alphad                             #Column Relative Humidity Tropics
           GammaM = MoistGamma(SST_TR_n, p_n)
           zt = (L/cp) * (qs(SST_TR_n, p_n) - q_top) / ( CRH * (GammaD - GammaM))       #Height of tropical domain
           TempTop = SST_TR_n - (L/cp) * qs(SST_TR_n, p_n) * ( (GammaD/ (CRH * (GammaD - GammaM))) - 1)      #Temperature at top of tropics
           
           Gamma0 = (SST_TR_n - TempTop) / zt                                  #Lapse Rate of tropical domain
           #print((TempTop/SST_TR_n)**(g/(Gamma0*Rv)))
           pt = p_n * (TempTop / SST_TR_n) ** (g / (Gamma0 * Rv))    #pressure at top of tropical domain
               
           qtop = qs(TempTop, pt)

        

        for jx in range(0,4):
            E = rho(SST_TR_n, p_n) * v * ct * qs(SST_TR_n, p_n) * (1-CRH)
            P = E + M * (q_n - q_top) / Sigma_TR_SFC                      #Moisture budget of tropics, solved for precipitation rate
            
            CRH = np.log( P / P0) / alphad                           #Column Relative Humidity Tropics
            zt = (SST_TR_n - TempTop) / (GammaD - CRH * (GammaD - GammaM))
            TempTop = SST_TR_n - (L/cp) * (qs(SST_TR_n, p_n) - q_top) * ( (GammaD/ (CRH * (GammaD - GammaM))) - 1)
            Gamma0 = (SST_TR_n - TempTop) / zt                                  #Lapse Rate of tropical domain

            
            #print((TempTop / SST_TR_n) ** (g / (Gamma0 * Rv)))
            pt = p_n * (TempTop / SST_TR_n) ** (g / (Gamma0 * Rv))    #pressure at top of tropical domain
                
            qtop = qs(TempTop, pt)

            Wmax = F(SST_TR_n) / Gamma0    #Total Possible amount of water vapor in tropics
            W_TR = CRH * Wmax              #Total water vapor in tropics
            
            
        q_up = q_top*10              #Moisture transferred from tropics to subtropics, accounting for ice in anvil cloud
        
        q_S_TR = CRH * qs(SST_TR_n, p_n)
        h_S_TR = cp * SST_TR_n + CRH * L * q_S_TR   
        
        
        
        #Tropical radiation solved 
        #SW Radiation
        l= CREFactor(CRH)                      #Assume Cloud Radiative Effects follow the form 
        CREPostFactor = l                                                      #    y = a(e^(bx^2)-1) for b = ln(2), a is emperically determined. 
                                                                #Look at Commonfunctions module for more.
         
        
        J_TR = Trans_TR(W_TR, CO2level)                         #Transmissivity of Tropics
        
        SW_CRE_SFC = alphaSWSFC * CREPostFactor                 #CRE of SFC and TOA
        SW_CRE_TOA = alphaSWTOA * CREPostFactor
        
        ACRE_SW = SW_CRE_TOA - SW_CRE_SFC                       #Atmospheric CRE of tropical column
        
        S_SFC_Dn_CLR_TR = S_TOA_Dn_TR * (1 - OzoneA) * J_TR     
        
        S_SFC_Up_CLR_TR = S_SFC_Dn_CLR_TR * albss
        S_SFC_Net_CLR_TR = S_SFC_Dn_CLR_TR - S_SFC_Up_CLR_TR
        S_SFC_Net_TR = S_SFC_Net_CLR_TR + SW_CRE_SFC
        
        S_TOA_Up_CLR_TR = S_TOA_Dn_TR * (1 - OzoneA) * albss * J_TR**2 
        
        S_TOA_Net_CLR_TR = S_TOA_Dn_TR - S_TOA_Up_CLR_TR
        S_TOA_Net_TR =  S_TOA_Net_CLR_TR + SW_CRE_TOA
        
        
        #LW Radiation 
        
        R_SFC_Up_TR = sb * SST_TR_n**4
        R_SFC_Dn_CLR_TR = R_SFC_Up_TR * f3TR(W_TR, CO2level)
        
        R_TOA_CLR_TR = R_SFC_Up_TR * f1TR(W_TR, CO2level)
        
        LW_CLRSKY_FLUX_DIFF_TR = R_SFC_Up_TR - R_SFC_Dn_CLR_TR - R_TOA_CLR_TR
        
        LW_CRE_SFC = alphaLWSFC * CREPostFactor
        LW_CRE_TOA = alphaLWTOA * CREPostFactor
        
        ACRE_LW = LW_CRE_TOA - LW_CRE_SFC
        
        LW_ALLSKY_FLUX_DIFF_TR = LW_CLRSKY_FLUX_DIFF_TR + ACRE_LW
        
        R_TOA_Net_TR = R_TOA_CLR_TR - LW_CRE_TOA
        
        R_SFC_Net_TR = R_SFC_Up_TR - R_SFC_Dn_CLR_TR - LW_CRE_SFC
        
        
        
        pb = p_n - deltapm_n                                                #pressure at BL top
        Ts = (h_n - L * q_n) / cp                                           #Temperature just above sea surface
        Gamma_ST = GammaD                                                   #Lapse Rate of the Subtropics
        zb_n = (Ts / Gamma_ST) * (1 - (pb / p_n) ** (Gamma_ST * Rd / g))    #first guess at Depth of boundary layer
        Tb = (h_n - g * zb_n - L * q_n) / cp                                #first guess at Temp at top of BL
        
        for nx in range (0, 4):

            qb = qs( Tb, pb)                                                #q_sat, i.e. moisture at top of BL
            
            if q_n < qb : #No Cloud Condition                               #Find cloud condition, which gives us idea of lapse rate
                 Gamma_ST = GammaD
            else:  #Cloud Condition
                 Gamma_ST = (Ts - Tb) / zb_n
            
            zb_n = (Ts / Gamma_ST) * (1 - (pb / p_n) ** (Gamma_ST * Rd / g))    #Second guess at depth of boundary layer
            Tb = (h_n - g * zb_n - L * q_n) / cp                                #Second guess at Temp of BL
        
        lb = max( q_n - qb, 0)                                                  #Liquid water at top of BL after condensation
        zc = zb_n * (1 - min( lb / (lb - (q_n - qs(Ts, p_n) ) ) , 1) )          #Predict Cloud base level
        Tc = - GammaD * zc + Ts                                                 #Temp at cloud base
        pc = p_n - rho(Ts, p_n) * g * zc                                        #Pressure at cloud base
        LWP = 0.5 * lb * (zb_n - zc) * (rho(Tb, pb) + rho(Tc, pc)) * 0.5        #Liquid water path calculation
        avgrho = (rho(SST_ST_n, p_n) + rho(Tb, pb)) / 2
        
        #albedo of Stratocumulus cloud
        ac = albmax*(1 - np.exp(-k*LWP))
        if zc >= zb_n:
            ac = 0
            
        qbplus = q_up                                                           #Moisture just above top of BL, same as moisture sent from tropics
        S_T = cp * TempTop + g * zt                                             #Dry static energy at top of tropics, same at top of subtropics due to WTG
        sbplus = S_T - Gamma0 * cp * (zt - zb_n)                                #Dry static energy just above BL top
        
        
            
            
            
        
            

        #Subtropical Radiation routine
        
        
        W_FT = qbplus * (pb - pt)/g                                             #Water vapor in free troposphere
        W_B = q_n * (p_n - pb)/g                                                #Water vapor in BL
        W_ST = W_FT + W_B                                                       #Water vapor in subtropics
        
        
            
        #Solar Radiation through clouds and on surface
        J_FT = Trans_ST(W_FT, CO2level)
        J_BL = Trans_ST(W_B, CO2level)
        J_ST = Trans_ST(W_ST, CO2level)
        
        #Downward clear sky solar
        S_TOA_Dn_CLR_ST = S_TOA_Dn_ST * (1- OzoneA)
        S_BLplus_Dn_CLR_ST = S_TOA_Dn_CLR_ST * J_FT
        S_SFC_Dn_CLR_ST = S_BLplus_Dn_CLR_ST * J_BL
        
        #Upward clear sky solar
        S_SFC_Up_CLR_ST = S_SFC_Dn_CLR_ST * albss
        S_BLplus_Up_CLR_ST = S_SFC_Up_CLR_ST * J_BL
        S_TOA_Up_CLR_ST = S_BLplus_Up_CLR_ST * J_FT
        
        #Net clear sky solar
        S_SFC_Net_CLR_ST = S_SFC_Dn_CLR_ST - S_SFC_Up_CLR_ST
        S_BLplus_Net_CLR_ST = S_BLplus_Dn_CLR_ST - S_BLplus_Up_CLR_ST
        S_TOA_Net_CLR_ST = S_TOA_Dn_CLR_ST - S_TOA_Up_CLR_ST
        S_Inf_Net_CLR_ST = S_TOA_Dn_ST - S_TOA_Up_CLR_ST
        
        
        #Downward full sky solar
        S_BLplus_Dn_ST = S_BLplus_Dn_CLR_ST
        S_SFC_Dn_ST = S_BLplus_Dn_ST * J_BL * (1 - ac) * (1 + 
                       ac * albss * J_BL**2)
        
        #Upward full sky solar
        S_SFC_Up_ST = S_SFC_Dn_ST * albss
        S_BLplus_Up_ST = S_BLplus_Dn_ST * (ac + albss * (J_BL**2) * (1 - ac)**2)
        
        S_TOA_Up_ST = S_BLplus_Up_ST * J_FT
        
        #Net full sky solar
        S_SFC_Net_ST = S_SFC_Dn_ST - S_SFC_Up_ST
        S_BLplus_Net_ST = S_BLplus_Dn_ST - S_BLplus_Up_ST
        S_TOA_Net_ST = S_TOA_Dn_ST - S_TOA_Up_ST
        
        S_Flux_Diff_FT_ST = S_TOA_Net_ST - S_BLplus_Net_ST
        
        #LW Radiation
        
        R_SFC_Up_ST = sb * (SST_ST_n**4) 
        R_SFC_Dn_CLR_ST = R_SFC_Up_ST * f3ST(W_ST, CO2level)
        R_TOA_CLR_ST = R_SFC_Up_ST * f1ST(W_ST, CO2level)
        R_BL_Up_CLR_ST = R_SFC_Up_ST * f1ST(W_B, CO2level)
        R_BLplus_Up_CLR_ST = R_BL_Up_CLR_ST
        R_BLplus_Up_ST = sb * (Tb**4) * (1 - np.exp(-k * LWP)) + R_BLplus_Up_CLR_ST * np.exp(-k * LWP)
        R_BLplus_Dn_ST  = R_SFC_Dn_CLR_ST * W_FT / W_ST
        
        R_BL_Dn_ST = R_BLplus_Dn_ST * np.exp( - k * LWP)
        R_BLplus_Net_ST = R_BLplus_Up_ST - R_BLplus_Dn_ST 
        R_BL_Up_ST = R_BL_Up_CLR_ST * np.exp(-k * LWP)
        R_SFC_Net_CLR_ST = R_SFC_Up_ST - R_SFC_Dn_CLR_ST
        R_BL_Net_ST = R_BL_Up_ST - R_BL_Dn_ST
        
        DeltaR = R_BLplus_Net_ST - R_BL_Net_ST
        R_SFC_Dn_ST = R_BLplus_Dn_ST * np.exp(-k * LWP) + sb * (Tc**4) * (1 - np.exp(-k * LWP))
        R_TOA_Net_ST = R_TOA_CLR_ST * np.exp(-k * LWP) + sb * (Tb**4) * (1 - np.exp(-k * LWP)) * f1ST(W_FT,CO2level)
        R_SFC_Net_ST = R_SFC_Up_ST - R_SFC_Dn_ST
        
        
        deltaql = qbplus - q_n
        hbplus = sbplus + L * qbplus
        deltah = hbplus - h_n
        Tbplus = (hbplus - g * zb_n - L*qbplus)/cp
        
        
        
        Shear = cd * (v**3)
        
        #######################################################################
        #subroutine for G0 and G1 and entrainment!
        #Subroutine is a linear interpolation between a guess for entrainment being zero and 0.1.
        #We find the slope and use this to find G1, which we then use to find the true entrainment
        
        W_E0 = 0
        E_n = W_E0 * avgrho
        gamma = ((L / Tb)**2) * epsilon * ClausClap(Tb) / (pb * cp * Rv)    #gamma and beta Taken from Dave's paper
        Beta = (1 + (1 + delta) * eps2 * gamma) / (gamma + 1)
        
        Fhs = rho(SST_ST_n, p_n) * v * ct * (cp * SST_ST_n + L * qs(SST_ST_n,p_n) - h_n) #Surface MSE flux, W/m^2
       
        Fhb = -E_n * deltah + DeltaR #Cloud top MSE flux, W/m^2        
                    
        Fqs = rho(SST_ST_n, p_n)*v*ct*(qs(SST_ST_n,p_n) - q_n)  #Moisture Flux at surface        
        
        Fqb = -E_n * deltaql #Moisture Flux at Cloud top    
        
        #Buoyancy Flux Calculation      
        
        Fhz0 = Fhs  
        Fqz0 = Fqs
        Fsvz0 = Fhz0 - (1 - delta * eps2) * L * Fqz0
        
        Fhzc = Fhs + (zc / zb_n) * (Fhb - Fhs)
        Fqzc = Fqs + (zc / zb_n) * (Fqb - Fqs)
        Fsvznoc = Fhzc - (1 - delta * eps2) * L * Fqzc
        Fsvzc = Fhzc * Beta - eps2 * L * Fqzc
        
        Fhzb = Fhb
        Fqzb = Fqb
        Fsvznob = Fhzb - (1 - delta * eps2) * L * Fqzb
        Fsvzb = Beta * Fhzb - eps2 * L * Fqzb
        
    
        
        #Buoyancy Flux integration
        if zc < zb_n:
            
            #With Cloud
           B_E0 = (g / (2 *rho(Ts, p_n) * cp )) * (zc * (Fsvz0 / Ts + Fsvznoc / Tc) 
                         + (zb_n - zc) * (Fsvzc / Tc + Fsvzb / Tb))
            
        else:
            
           #Without Cloud
            B_E0 = (g / (2 * rho(Ts, p_n) * cp ) ) * (zc * (Fsvz0 / Ts + Fsvznoc / Tc)
                            + (zb_n - zc) * (Fsvznoc / Tc + Fsvznob / Tb) )
                
        

        #G0 represents energy flux from BL due to Shear and buoyant air rising with no entrainment
        G0 = B_E0 + Shear
        
        #Now guess a new nonzero entrainment velocity
        W_E1 = 0.1
        E_n = W_E1 * avgrho
        
        Fhs = rho(SST_ST_n, p_n) * v * ct * (cp * SST_ST_n + L * qs(SST_ST_n,p_n) - h_n) #Surface MSE flux, W/m^2
       
        Fhb = -E_n * deltah + DeltaR #Cloud top MSE flux, W/m^2        
                    
        Fqs = rho(SST_ST_n, p_n)*v*ct*(qs(SST_ST_n,p_n) - q_n)    #Moisture Flux at surface        
        
        Fqb = -E_n * deltaql #Moisture Flux at Cloud top    
        
        #Buoyancy Flux Calculation
        
        Fhz0 = Fhs  
        Fqz0 = Fqs
        Fsvz0 = Fhz0 - (1 - delta * eps2) * L * Fqz0
        
        Fhzc = Fhs + (zc / zb_n) * (Fhb - Fhs)
        Fqzc = Fqs + (zc / zb_n) * (Fqb - Fqs)
        Fsvznoc = Fhzc - (1 - delta * eps2) * L * Fqzc
        Fsvzc = Fhzc * Beta - eps2 * L * Fqzc
        
        Fhzb = Fhb
        Fqzb = Fqb
        Fsvznob = Fhzb - (1 - delta * eps2) * L * Fqzb
        Fsvzb = Beta * Fhzb - eps2 * L * Fqzb
        
        #Buoyancy Flux integration
        if zc < zb_n:
            
            #With Cloud
           B_E1 = (g / (2 *rho(Ts, p_n) * cp )) * (zc * (Fsvz0 / Ts + Fsvznoc / Tc) 
                         + (zb_n - zc) * (Fsvzc / Tc + Fsvzb / Tb))
            
        else:
            
           #Without Cloud
            B_E1 = (g / (2 * rho(Ts, p_n) * cp ) ) * (zc * (Fsvz0 / Ts + Fsvznoc / Tc)
                            + (zb_n - zc) * (Fsvznoc / Tc + Fsvznob / Tb) )
     
        G1 = (B_E0 - B_E1) / (W_E0 - W_E1)
        
        q = -ce0 * G0
        p = -ce1 * G1
        
        #Cardano's formula
        sqrt = np.sqrt( (q/2)**2 + (p/3)**3)
        
        W_E = np.cbrt( -q/2 + sqrt) + np.cbrt( -q/2 - sqrt)
        
        
        #################################################################### End subroutine
        
        
        #Actual entrainment rate!
        E_n = W_E * avgrho
        
        Fhs = rho(SST_ST_n, p_n) * v * ct * (cp * SST_ST_n + L * qs(SST_ST_n,p_n) - h_n) #Surface MSE flux, W/m^2
       
        Fhb = -E_n * deltah + DeltaR #Cloud top MSE flux, W/m^2        
                    
        Fqs = rho(SST_ST_n, p_n)*v*ct*(qs(SST_ST_n,p_n) - q_n)    #Moisture Flux at surface        
        
        Fqb = -E_n * deltaql #Moisture Flux at Cloud top    
        
        
        #Buoyancy Flux Calculation
        
        Fhz0 = Fhs  
        Fqz0 = Fqs
        Fsvz0 = Fhz0 - (1 - delta * eps2) * L * Fqz0
        
        Fhzc = Fhs + (zc / zb_n) * (Fhb - Fhs)
        Fqzc = Fqs + (zc / zb_n) * (Fqb - Fqs)
        Fsvznoc = Fhzc - (1 - delta * eps2) * L * Fqzc
        Fsvzc = Fhzc * Beta - eps2 * L * Fqzc
        
        Fhzb = Fhb
        Fqzb = Fqb
        Fsvznob = Fhzb - (1 - delta * eps2) * L * Fqzb
        Fsvzb = Beta * Fhzb - eps2 * L * Fqzb
        
        sv_min_sv_crit = Beta * deltah - eps2 * L *deltaql
        
        #Buoyancy Flux integration
        if zc < zb_n:
            
            #With Cloud
           B_E = (g / (2 *rho(Ts, p_n) * cp )) * (zc * (Fsvz0 / Ts + Fsvznoc / Tc) 
                         + (zb_n - zc) * (Fsvzc / Tc + Fsvzb / Tb))
            
        else: 
            
           #Without Cloud
            B_E = (g / (2 * rho(Ts, p_n) * cp ) ) * (zc * (Fsvz0 / Ts + Fsvznoc / Tc)
                            + (zb_n - zc) * (Fsvznoc / Tc + Fsvznob / Tb) )
       
#############################################################################################
        #Coupling Mechanism
        
        rho_top = rho(TempTop, pt)
        num_eval_points = 100                                #discretize integral for coupler
        Gamma = GammaD
        zbplus = 1 + (Gamma / TempTop) * (zt - zb_n)
        z_n = np.linspace(1, zbplus, num_eval_points)
        
        if timestep == 0:
            Sigma_ST_top = 0.85
        else:
            Sigma_ST_top = Sigma_ST_zb * rho_layer(zb_n , rho_top, Gamma, TempTop, zt) / rho_top
            
        Int_Q_ST_dz = R_BLplus_Net_ST - R_TOA_Net_ST + S_Flux_Diff_FT_ST
        Q_ST = Int_Q_ST_dz * g / (pb - pt)
        
        Int_Q_TR_dz = - (rho_top * Sigma_ST_top * g / (pb - pt)) * Int_Q_ST_dz * \
            Integral_zbplus_1(z_n, num_eval_points, Gamma, TempTop, zt, zb_n, Sigma_ST_top)
        Q_TR = Int_Q_TR_dz * g / (pb - pt)
            
        dsdz = -cp * Gamma + g
        
        M = -rho_top * Sigma_ST_top * g * Int_Q_ST_dz / (dsdz * (pb - pt))
        
        Sigma_ST_zb = rho_top * Sigma_ST_top / rho_layer( zb_n , rho_top, Gamma, TempTop, zt ) 
        Sigma_ST_ss = rho_top * Sigma_ST_top / rho_layer( 0, rho_top, Gamma, TempTop, zt)
        if M <= 0:
            M = 0
#############################################################################
            
        
        q_dn = q_S_TR * Sigma_ST_ss + q_n * (1 - Sigma_ST_ss)    #Calculate Moisture taken by M from subtropics to tropics
        h_dn = h_S_TR * Sigma_ST_ss + h_n * (1 - Sigma_ST_ss)    #Calculate MSE taken by M from subtropics to tropics_
        
        Fhe = E_n * deltah                        #Energy from entrained air across the boundary layer.
        Fhr = R_SFC_Net_ST - R_BLplus_Net_ST      #Energy gained by boundary layer from sfc to BL top
        Fhsolar = S_SFC_Net_ST - S_BLplus_Net_ST  #Energy gained by BL from SFC to BL top
        Fqe = E_n * deltaql                       #Moisture Loss from entrained air across BL
        FhM = M * (h_n - h_S_TR)                  #Energy transfer in BL by M
        FqM = M * (q_n - q_S_TR)                  #Moisture transfer in BL by M
        
        
        #Advance Moist Static Energy and Moisture in boundary layer
        h_n = h_n + g * tstep * (FhM + Fhs + Fhe + Fhr - Fhsolar) / deltapm_n
        q_n = q_n + g * tstep * (FqM + Fqs + Fqe ) / deltapm_n
        
        
        #Advance pressure depth of Boundary Layer
        deltapm_n = deltapm_n + tstep * g * (E_n - M / Sigma_ST_zb)
        
        
        #Advance SST in Tropics and Subtropics
        # SST_ST_n = SST_ST_n + tstep*(S_SFC_Net_ST - R_SFC_Net_ST - Fhs)/(rhowater*c*D)
        # SST_TR_n = SST_TR_n + tstep * (S_SFC_Net_TR - R_SFC_Net_TR - L * E) / (rhowater*c*D)
        
        logging.debug(f'Finished all calcs in timstep {timestep}')
        
        if timestep % 5 == 0:
            i = timestep // steps_per_collection
            
            times[i] = timescale[timestep]
            hms[i] = h_n
            qms[i] = q_n
            deltapms[i] = deltapm_n
            SSTs[i] = SST_ST_n
            zbs[i] = zb_n
            
            pbs[i] = pb
            zcs[i] = zc
            LWPs[i] = LWP
            
            Tcs[i] = Tc
            svs[i] = sv_min_sv_crit
            B_Es[i] = B_E 
            Ents[i] = E_n
            W_Es[i] = W_E
            G0s[i] = G0
            G1s[i] = G1
            
            
            qbs[i] = qb
            lbs[i] = lb
            Tbs[i] = Tb
            deltaqls[i] = deltaql
            deltahs[i] = deltah
            hbpluses[i] = hbplus
            Tbpluses[i] = Tbplus
            qbpluses[i] = qbplus
            sbpluses[i] = sbplus
            avgrhos[i] = avgrho
            
            
              
           
            Fhss[i] = Fhs
            Fqss[i] = Fqs
            Fhbs[i] = Fhb
            Fqbs[i] = Fqb
            Fhes[i] = Fhe
            Fhrs[i] = Fhr 
            Fhsolars[i] = Fhsolar 
            Fqes[i] = Fqe 
            FhMs[i] = FhM 
            FqMs[i] = FqM 
            Shears[i] = Shear 
          
            
            #Subtropical Radiation Lists
            #Net LW ST Radiation
            DeltaRs[i] = DeltaR 
            R_SFC_Net_STs[i] = R_SFC_Net_ST 
            R_BL_Net_STs[i] = R_BL_Net_ST 
            R_BLplus_Net_STs[i] = R_BLplus_Net_ST 
            R_TOA_Net_STs[i] = R_TOA_Net_ST 
            R_SFC_Net_CLR_STs[i] = R_SFC_Net_CLR_ST 
            
            #Upward LW Radiation
            R_SFC_Up_STs[i] = R_SFC_Up_ST 
            R_TOA_CLR_STs[i] = R_TOA_CLR_ST 
            R_BL_Up_CLR_STs[i] = R_BL_Up_CLR_ST 
            R_BLplus_Up_CLR_STs[i] = R_BLplus_Up_CLR_ST 
            R_BLplus_Up_STs[i] = R_BLplus_Up_ST 
            R_BL_Up_STs[i] = R_BL_Up_ST 
            
            #Downward LW Radiation
            R_SFC_Dn_CLR_STs[i] = R_SFC_Dn_CLR_ST 
            R_SFC_Dn_STs[i] = R_SFC_Dn_ST 
            R_BLplus_Dn_STs[i]  = R_BLplus_Dn_ST 
            R_BL_Dn_STs[i] = R_BL_Dn_ST    
            
            
            
            #Water vapor in each layer and occuring transmissivity of layer
            W_FTs[i] = W_FT 
            W_Bs[i] = W_B 
            W_STs[i] = W_ST 
            J_FTs[i] = J_FT 
            J_BLs[i] = J_BL 
            J_STs[i] = J_ST 
            
            #Downward clear sky solar
            S_TOA_Dn_CLR_STs[i] = S_TOA_Dn_CLR_ST 
            S_BLplus_Dn_CLR_STs[i] = S_BLplus_Dn_CLR_ST 
            S_SFC_Dn_CLR_STs[i] = S_SFC_Dn_CLR_ST 
            
            #Upward clear sky solar
            S_SFC_Up_CLR_STs[i] = S_SFC_Up_CLR_ST 
            S_BLplus_Up_CLR_STs[i] = S_BLplus_Up_CLR_ST 
            S_TOA_Up_CLR_STs[i] = S_TOA_Up_CLR_ST 
            
            #Net clear sky solar
            S_SFC_Net_CLR_STs[i] = S_SFC_Net_CLR_ST 
            S_BLplus_Net_CLR_STs[i] = S_BLplus_Net_CLR_ST 
            S_TOA_Net_CLR_STs[i] = S_TOA_Net_CLR_ST 
            S_Inf_Net_CLR_STs[i] = S_Inf_Net_CLR_ST    
            
            #Downward full sky solar
            S_BLplus_Dn_STs[i] = S_BLplus_Dn_ST 
            S_SFC_Dn_STs[i] = S_SFC_Dn_ST 
            S_TOA_Dn_STs[i] = S_TOA_Dn_ST 
            
            #Upward full sky solar
            S_SFC_Up_STs[i] = S_SFC_Up_ST 
            S_BLplus_Up_STs[i] = S_BLplus_Up_ST 
            S_TOA_Up_STs[i] = S_TOA_Up_ST 
            
            #Net full sky solar
            S_SFC_Net_STs[i] = S_SFC_Net_ST 
            S_BLplus_Net_STs[i] = S_BLplus_Net_ST 
            S_TOA_Net_STs[i] = S_TOA_Net_ST
            S_Flux_Diff_FT_STs[i] = S_Flux_Diff_FT_ST
            
            
            #Tropical Lists
            Es[i] = E 
            Ps[i] = P 
            CRHs[i] = CRH 
            zts[i] = zt 
            TempTops[i] = TempTop 
            Gamma0s[i] = Gamma0 
            pts[i] = pt 
            Wmaxes[i] = Wmax 
            W_TRs[i] = W_TR 
            q_tops[i] = q_top 
            q_ups[i] = q_up 
            q_S_TRs[i] = q_S_TR 
            h_S_TRs[i] = h_S_TR
            
            #Tropical Radiation lists
            
            J_TRs[i] = J_TR 
                #Cloud effects
            SW_CRE_SFCs[i] = SW_CRE_SFC
            SW_CRE_TOAs[i] = SW_CRE_TOA
            ACRE_SWs[i] = ACRE_SW 
            
                #Clr Sky
            S_SFC_Dn_CLR_TRs[i] = S_SFC_Dn_CLR_TR
            S_SFC_Up_CLR_TRs[i] = S_SFC_Up_CLR_TR
            S_SFC_Net_CLR_TRs[i] = S_SFC_Net_CLR_TR
            S_TOA_Up_CLR_TRs[i] = S_TOA_Up_CLR_TR
            S_TOA_Net_CLR_TRs[i] = S_TOA_Net_CLR_TR
            
                #Net Radiation at Top and Bottom
            S_SFC_Net_TRs[i] = S_SFC_Net_TR
            S_TOA_Net_TRs[i] = S_TOA_Net_TR        
            
                #LW Radiation 
            R_SFC_Up_TRs[i] = R_SFC_Up_TR
            R_SFC_Dn_CLR_TRs[i] = R_SFC_Dn_CLR_TR
            R_TOA_CLR_TRs[i] = R_TOA_CLR_TR
        
            LW_CRE_SFCs[i] = LW_CRE_SFC
            LW_CRE_TOAs[i] = LW_CRE_TOA
            ACRE_LWs[i] = ACRE_LW
            LW_ALLSKY_FLUX_DIFF_TRs[i] = LW_ALLSKY_FLUX_DIFF_TR
            LW_CLRSKY_FLUX_DIFF_TRs[i] = LW_CLRSKY_FLUX_DIFF_TR
            
            R_TOA_Net_TRs[i] = R_TOA_Net_TR
            R_SFC_Net_TRs[i] = R_SFC_Net_TR
            
          
            #Coupling Lists
            
            Int_Q_TR_dzs[i] = Int_Q_TR_dz
            Q_TRs[i] = Q_TR
            Int_Q_ST_dzs[i] = Int_Q_ST_dz 
            Q_STs[i] = Q_ST
            
            Sigma_TRs[i] = 1 - Sigma_ST_ss
            Sigma_STs[i] = Sigma_ST_ss
            Ms[i] = M
            q_dns[i] = q_dn
            
        
        
        if timestep // 5 >= len(times):
            break
        
        
        
        logging.debug(f'End of timestep {timestep}')
        
    
    time_da = xr.DataArray(
        data = times, 
        dims = ["time"],
        coords = {"time":times},
        attrs = {"long_name":"Model Timestep","units":"Timesteps from model initialization"}
    )
    
    
    
    # Actually run the model calculations, and update the variables in the dstore dictionary with statements like
    dstore = dict({"h_m":hms, "ql_m":qms, "deltap_m":deltapms, "SST_ST":SSTs, "SST_TR":SST_TRs, 
                   "z_B":zbs, "p_B":pbs, "z_C":zcs, "LWP":LWPs, 

     		 "T_c":Tcs, "deltas_v_minus_deltas_v_crit":svs, "B_Flux":B_Es, "Ent":Ents, "W_E":W_Es, "G0":G0s, "G1":G1s,

    		 "q_B":qbs, "l_B":lbs, "T_B":Tbs, "delta(q+l)":deltaqls, "deltah":deltahs, "h_B+":hbpluses, "T_B+":Tbpluses,
    		 "AVG BL Rho":avgrhos,

    		 "Fh_ss":Fhss, "Fq_ss":Fqss, "Fh_B":Fhbs, "Fq_B":Fqbs, "Fh_E":Fhes, "Fh_R":Fhrs, "Fh_Solar":Fhsolars, 
    		 "Fq_E":Fqes, "Fh_M":FhMs, "Fq_M":FqMs, "Shear":Shears,

    		 "DeltaR":DeltaRs, "R_SFC_Net_ST":R_SFC_Net_STs, "R_BL_Net_ST":R_BL_Net_STs, "R_BL+_Net_ST":R_BLplus_Net_STs, 
    		 "R_TOA_Net_ST":R_TOA_Net_STs, "R_SFC_Net_CLR_ST":R_SFC_Net_CLR_STs, 

    		 "R_SFC_Up_ST":R_SFC_Up_STs, "R_TOA_CLR_ST":R_TOA_CLR_STs, "R_BL_Up_CLR_ST":R_BL_Up_CLR_STs, 
             "R_BL+_Up_CLR_ST":R_BLplus_Up_CLR_STs, "R_BL+_Up_ST":R_BLplus_Up_STs, "R_BL_Up_ST":R_BL_Up_STs, 

    		 "R_SFC_Dn_CLR_ST":R_SFC_Dn_CLR_STs, "R_SFC_Dn_ST":R_SFC_Dn_STs, "R_BL+_Dn_ST":R_BLplus_Dn_STs, 
             "R_BL_Dn_ST":R_BL_Dn_STs,

    		 "W_FT":W_FTs, "W_B":W_Bs, "W_ST":W_STs, "J_FT":J_FTs, "J_BL":J_BLs, "J_ST":J_STs,

    		 "S_TOA_Dn_CLR_ST":S_TOA_Dn_CLR_STs, "S_BL+_Dn_CLR_ST":S_BLplus_Dn_CLR_STs, "S_SFC_Dn_CLR_ST":S_SFC_Dn_CLR_STs,

    		 "S_SFC_Up_CLR_ST":S_SFC_Up_CLR_STs, "S_BL+_Up_CLR_ST":S_BLplus_Up_CLR_STs, "S_TOA_Up_CLR_ST":S_TOA_Up_CLR_STs, 

    		 "S_SFC_Net_CLR_ST":S_SFC_Net_CLR_STs, "S_BL+_Net_CLR_ST":S_BLplus_Net_CLR_STs, 
             "S_TOA_Net_CLR_ST":S_TOA_Net_CLR_STs, "S_Inf_Net_CLR_ST":S_Inf_Net_CLR_STs, 

    		 "S_BL+_Dn_ST":S_BLplus_Dn_STs, "S_SFC_Dn_ST":S_SFC_Dn_STs, "S_TOA_Dn_ST":S_TOA_Dn_STs, 

    		 "S_SFC_Up_ST":S_SFC_Up_STs, "S_BL+_Up_ST":S_BLplus_Up_STs, "S_TOA_Up_ST":S_TOA_Up_STs,  

    		 "S_SFC_Net_ST":S_SFC_Net_STs, "S_BL+_Net_ST":S_BLplus_Net_STs, "S_TOA_Net_STs":S_TOA_Net_STs,
             "S_Flux_Diff_FT_ST":S_Flux_Diff_FT_STs,
    	
    		 "Evap":Es, "Precip":Ps, "CRH":CRHs,  "Z_T":zts, "T_T":TempTops, "Gamma_0":Gamma0s,
             "P_T":pts, "W_Max":Wmaxes, "W_TR":W_TRs, "q_T":q_tops, "q_up":q_ups, "q_S_TR":q_S_TRs, "h_S_TR":h_S_TRs, 
    		 
    		 "J_TR":J_TRs, "SW_CRE_SFC":SW_CRE_SFCs, "SW_CRE_TOA":SW_CRE_TOAs, "ACRE_SW":ACRE_SWs,

    		 "S_SFC_Dn_CLR_TR":S_SFC_Dn_CLR_TRs, "S_SFC_Up_CLR_TR":S_SFC_Up_CLR_TRs, "S_SFC_Net_CLR_TR":S_SFC_Net_CLR_TRs, 
    		 "S_TOA_Up_CLR_TR":S_TOA_Up_CLR_TRs, "S_TOA_Net_CLR_TR":S_TOA_Net_CLR_TRs,
    	
    		 "S_SFC_Net_TR":S_SFC_Net_TRs, "S_TOA_Net_TR":S_TOA_Net_TRs,
    		
    		 "R_SFC_Up_TR":R_SFC_Up_TRs, "R_SFC_Dn_CLR_TR":R_SFC_Dn_CLR_TRs, "R_TOA_CLR_TR":R_TOA_CLR_TRs,

    		 "LW_CRE_SFC":LW_CRE_SFCs, "LW_CRE_TOA":LW_CRE_TOAs, "ACRE_LW":ACRE_LWs, 
             "LW_ALLSKY_FD_TR":LW_ALLSKY_FLUX_DIFF_TRs, "LW_CLRSKY_FD_TR":LW_CLRSKY_FLUX_DIFF_TRs,

    		 "R_TOA_Net_TR":R_TOA_Net_TRs, "R_SFC_Net_TR":R_SFC_Net_TRs,

    		 "Int_Q_TR_dz":Int_Q_TR_dzs, "Q_TR":Q_TRs, "Int_Q_ST_dz":Int_Q_ST_dzs, "Q_ST":Q_STs, 

    		 "Sigma_TR":Sigma_TRs, "Sigma_ST":Sigma_STs, "Mass Flux":Ms, "q_Dn":q_dns })


    dims   = ["time"]
    coords = {"time":time_da}
    
    save_to_disk(namelist_vars, dstore, save_metadata, coords, dims)
    
    
    return Ps

print(Main( 5, 304000, 0.008, 6000, 285, 100000, 305, "Present"))
        
        
            
        
        
        
            
        
        