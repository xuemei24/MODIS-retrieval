import numpy as np
import iris,glob
import iris.quickplot as qplt
from iris.coord_systems import GeogCS
import iris.analysis.cartography
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import LogNorm
from pyresample import kd_tree
from pyresample.plot import show_quicklook
from pyresample import geometry,area_config
from netCDF4 import Dataset
from numpy import ma
import math,string
import sys
import xarray as xr
from pylab import *
import os
import matplotlib as mpl
mpl.use('Agg')

# Script to regrid MODIS level 2 cloud and aerosol data and UM regional simulation data to the same grid for plotting and/or evaluation


# Satellite data is stored for day of year. e.g. 2018.03.22
# 2018 was a common year (as opposed to leap year)
# Derive the DOY for 2018.03.22
year = '2018'
doy_array = [0,31,59,90,120,151,181,212,243,273,304,334]
day= 22
month=3
doy = '0'+str(doy_array[month-1]+day) if doy_array[month-1]+day < 100 else str(doy_array[month-1]+day)

# Set the path to the MODIS files
monthstring = '0'+str(month) if month < 10 else str(month)
daystring = str(day)
modis_l2_init='MOD06_L2.A'
modis_path='/ocean/projects/atm200005p/xwangw/analysis/observations/modis/mod06_global/'
# Select all the files for 2018.03.22
modis_files = sorted(glob.glob(modis_path+modis_l2_init+year+str(doy)+'*.nc'))

# Regridded CDNC for each hour is stored here
path_to_save_cdnc_per_hour = '/ocean/projects/atm200005p/xwangw/analysis/saved_txt/model_run_saved_txt/regridded_cdnc/'

# Set the path to the UM output files
path_model1='/ocean/projects/atm200005p/xwangw/netcdfs/u-ct879/march/regional/201803'

# Find any 3 dimension cube (time,lat,lon) that can be used to store UM CDNC at cloud top height
path_pafile='/ocean/projects/atm200005p/xwangw/netcdfs/u-ct879/march/regional/20180318/umnsaa_pa000' 
cube3d='toa_outgoing_shortwave_flux'

# Define and check target grid - the grid we will present results on from both satellite and model
# First define a target grid at 1km resolution
area_def = geometry.AreaDefinition.from_extent(area_id='southern_ocean_1km',projection=ccrs.PlateCarree(),shape=(2882,7722),area_extent=(109.7,-61.4,179.9,-35.2),units='deg') 
lon,lat = area_def.get_lonlats()
print('check lon',lon.shape)
print('check lat',lat.shape)

# Second, define a target grid at 6 km resolution (Match the model resolution)
area_def2 = geometry.AreaDefinition.from_extent(area_id='southern_ocean_6km',projection=ccrs.PlateCarree(),shape=(480,1287),area_extent=(109.7,-61.4,179.9,-35.2),units='deg') 
lon2,lat2 = area_def2.get_lonlats()

# Read in the altitudes from the model output and store it to hr
a = Dataset('/ocean/projects/atm200005p/xwangw/netcdfs/u-ct879/march/regional/20180318/abl_umnsaa_pj000.nc')
hr = a.variables['level_height'][:]

# The following lines (until def regrid_modis_data) select MODIS granules for each hour in a day.
# I manually assigned the number of files for each hour (number_of_files). Otherwise, please refer
# to /ocean/projects/atm200005p/shared/python-analysis/resample_modis.py to select the granules.
# I also manually selected the corresponding UM output for each day (ppfiles)
number_of_files = {'0':7,'1':8,'2':3,'3':11,'4':4,'5':7,'6':8,'7':4,'8':10,'9':5,'10':6,'11':9,'12':3,'13':10,'14':6,'15':5,'16':10,'17':3,'18':9,'19':6,'20':5,'21':10,'22':4,'23':8}
ppfiles = [path_model1+'20/umnsaa_pj048',path_model1+'20/umnsaa_pj054',path_model1+'20/umnsaa_pj060',path_model1+'20/umnsaa_pj066']

cycle_len = 2                        # Days of cycle length
first_day_of_cycle = '18'            # The date of the first model cycle
first_day_of_the_second_cycle = '20' # The date of the second cycle (if there is one)

# Calculate CDNC from MODIS for each granule
def qsatw(T, p):
    Tzero=273.16
    T0 = np.copy(T)
    T0.fill(273.16)
    try:
        log10esw = (10.79574*(1-np.divide(T0,T))-5.028*np.log10(np.divide(T,T0))
                    + 1.50475e-4*(1-np.power(10,(-8.2369*(np.divide(T,T0)-1))))
                    + 0.42873e-3*(np.power(10,(4.76955*(1-np.divide(T,T0))))-1) + 2.78614)
    except Exception as e:
        print('str(e)=',str(e),'exception at T='+str(T))
    esw = 10**log10esw
    qsw = 0.62198*esw/(p-esw)

    return qsw

def dqdz(T, p):
    g=9.8
    cp = 1004.67 # J/kgK
    Hv = 1000.0*(2500.8-2.36*(T-273.15)+
                  0.0016*np.power((T-273.15),2) -
                  0.00006*np.power((T-273.15),3.0))
    Rsd = 287.04
    Rsw = 461.5
    #qsatw2 = np.vectorize(qsatw)
    r = qsatw(T, p)

    Gamma_w_num = g*(1+(Hv/Rsd)*np.divide(r,T))
    Gamma_w_den = (cp + (0.622*(Hv**2)/Rsd)*np.divide(r,np.power(T,2.0))) #K/m
    Gamma_w  = np.divide(Gamma_w_num,Gamma_w_den)
     # J/kg wikipedia from Rogers & Yau, a Short Course in Cloud Physics; approx 2.6e6
    cpa = np.copy(T)
    Gamma_d = np.copy(T)
    cpa.fill(cp)
    Gamma_d.fill(g/cp)
    Cw1 = np.divide(cpa,Hv)*(Gamma_d-Gamma_w) #Ahmad 2013 Tellus B
    # units: m^-1, so need to multiply by rho_a (Grosvenor)
    Cw = (p/Rsd)*np.divide(Cw1,T) # units kgm^-4
    return Cw

def getCDNC(df):
    k = 0.8 # Brenguier et al 2011; Martin et al 1994 apparently suggests 0.67 and Painemal & Zuidema 0.88, all for MBL. Grosvenor & Wood (2014) uses 0.8
    f = 0.7 # Grosvenor & Wood 2014
    Q=2
    rho_w = 1000 #kg m^-3
    #p = 850*100.0 #Pa      
    ctopp = df['Cloud_Top_Pressure']*100. #Pa
    nd_prefactor = (2*math.sqrt(10.0)/(k*math.pi*Q**3.0))
    effradius = df['Cloud_Effective_Radius_37']
    cot = df['Cloud_Optical_Thickness_37']
    lwp = df['Cloud_Water_Path_37']
    phase=df['Cloud_Phase_Infrared']
    temperature = df['Cloud_Top_Temperature']
    sza = df['Solar_Zenith']
    cth = df['Cloud_Top_Height']
    mcf = df['Cloud_Fraction']
    cldmask = df['Cloud_Mask_1km']

    latitude = df['latitude']
    longitude = df['longitude']

    Cw = dqdz(temperature,ctopp)

    print('nd_prefactor = '+str(nd_prefactor),'Cw',Cw)

    nd_sqrtarg_num = f*Cw*cot
    nd_sqrtarg_den = rho_w*np.power(1e-6*effradius,5.0) #Grosvenor & Wood; effradius is in microns
    nd1 = 1e-6*nd_prefactor*np.sqrt(np.divide(nd_sqrtarg_num,nd_sqrtarg_den)) # units cm^-3

    nd1 = np.where(~np.isnan(nd1),nd1,0)
    nd1 = np.where(~np.isinf(nd1),nd1,0)

    nd2 = np.where(phase<2,nd1,0)  # remove ice clouds, 0 -- cloud free, 1 -- water cloud, 2 -- ice cloud, 3 -- mixed phase cloud, 6 -- undetermined phase
    nd3 = np.where(lwp>2,nd2,0)    # remove thin clouds
    nd4 = np.where(cot>3,nd3,0)    # remove thin clouds
    nd5 = np.where(sza<60,nd4,0)   # remove the grids with solar zenith angles that are too high (sun is low)
    nd6 = np.where(mcf>0.8,nd5,0)  # remove the grids with cloud fraction < 0.8
    nd = nd6

    return nd,latitude,longitude

# CDNC from each file is stored
CDNC_per_file = {}
lat_rg = {}
lon_rg = {}

for ij,fn in enumerate(modis_files):
    df = xr.open_dataset(fn)
    Nd,lats0,lons0 = getCDNC(df)
    CDNC_per_file[str(ij)] = Nd
    lat_rg[str(ij)] = lats0
    lon_rg[str(ij)] = lons0
    print('ij',ij,'cdnc shape',Nd.shape)

print('CDNC_per_file.keys=',CDNC_per_file.keys())


# This subroutine reads in the MODIS data and regrids it
# The regridded results are a list of up to three numpy arrays, each covering the
# whole domain, but each masked and only corresponding to one swath. Choose unmasked 
# data points from each to combine into a single numpy array.
def regrid_modis_data(cdnc,lat_rg,lon_rg,number_of_files,regrid_method,var_name):
    nd_plot={}
    acc_file = 0
    for ij in number_of_files.keys():
        print('acc_file',acc_file,'ij',ij)
        if number_of_files[ij] == 0:
            nd_plot[ij]=[]
            print('regrid skip',ij)
            continue
        else:
            for k in range(number_of_files[ij]):
                print(acc_file+k,number_of_files[ij])
                nd_temp = cdnc[str(acc_file+k)]
                lat_temp = lat_rg[str(acc_file+k)]
                lon_temp = lon_rg[str(acc_file+k)]

                nd_temp = np.ma.masked_where(nd_temp<=0,nd_temp)
                swath_def = geometry.SwathDefinition(lons=lon_temp, lats=lat_temp)
                if k > 0:
                    big_swath_def = swath_def.concatenate(big_swath_def)
                    big_nd_array = np.concatenate((nd_temp,big_nd_array))
                else:
                    big_swath_def = swath_def
                    big_nd_array = nd_temp
                print('2nd shape',k,big_nd_array.shape, big_swath_def.shape)

            acc_file+=number_of_files[ij]

            if regrid_method ==1:
                # nearest-neighbour regrid is one of several options. This is sensible when source and target pixels are the same size. 
                # radius of influence is in meters, for each target pixel, will look up to (1+epsilon) x this many meters away for a source pixel - try 10km
                # fill_value=None will mask undetermined pixels
                # https://pyresample.readthedocs.io/en/latest/api/pyresample.html?highlight=concatenate#module-pyresample.kd_tree
                # MODIS data is first regridded to 1 km resolution
                result1km = kd_tree.resample_nearest(big_swath_def, big_nd_array, area_def, radius_of_influence=1000, epsilon=0.5, fill_value=None)
                # Then, the 1 km results are regridded to 6 km resolution to match the model output
                result6km = kd_tree.resample_nearest(geometry.SwathDefinition(lons=lon,lats=lat), result1km, area_def2, radius_of_influence=1000, epsilon=0.5, fill_value=None)

            elif regrid_method==2:
                #Gaussian weighting will use several neighbors, so probably makes more sense if target grids are a little larger than source grids
                result = kd_tree.resample_gauss(big_swath_def, big_nd_array, area_def, radius_of_influence=10000, sigmas=5000, fill_value=None)
            result6km = np.ma.masked_where(result6km<=0,result6km)

            print('3rd shape',len(result6km))
        nd_plot[ij]=result6km
    return nd_plot

# The regridded CDNC for each hour is stored and saved
CDNC_per_hour = regrid_modis_data(CDNC_per_file,lat_rg,lon_rg,number_of_files,1,'cdnc')
for ij in range(len(CDNC_per_hour)):
    print(CDNC_per_hour[str(ij)])
    if len(CDNC_per_hour[str(ij)])==0:
        print('skip')
        continue

    # Remove unrealistic CDNC
    CDNC_plot = np.where(CDNC_per_hour[str(ij)]>0,CDNC_per_hour[str(ij)],np.nan)
    CDNC_plot = np.where(CDNC_plot<5000,CDNC_plot,np.nan)
    CDNC_per_file[str(ij)] = CDNC_plot

    if int(ij)>=10:
        np.savetxt(path_to_save_cdnc_per_hour+'modis_cdnc_'+daystring+'th_'+str(ij)+'utc.txt', CDNC_plot, fmt='%.18e', delimiter=',')
    else:
        np.savetxt(path_to_save_cdnc_per_hour+'modis_cdnc_'+daystring+'th_0'+str(ij)+'utc.txt', CDNC_plot, fmt='%.18e', delimiter=',')

# Now find the CDNC at cloud top height in the model output
def get_air_density( air_pressure,potential_temperature):
  p0 = iris.coords.AuxCoord(1000.0,
                            long_name='reference_pressure',
                            units='hPa')
  p0.convert_units(air_pressure.units)

  Rd=287.05 # J/kg/K
  cp=1005.46 # J/kg/K
  Rd_cp=Rd/cp

  temperature=potential_temperature*(air_pressure/p0)**(Rd_cp)
  temperature._var_name='temperature'
  R_specific=iris.coords.AuxCoord(287.058,
                                  long_name='R_specific',
                                  units='J-kilogram^-1-kelvin^-1')#J/(kgK)

  air_density=(air_pressure/(temperature*R_specific))
  air_density.long_name='Density of air'
  air_density._var_name='air_density'
  air_density.units='kg m-3'
  temperature.units='K'
  return [air_density, temperature]

def load_with(filepath,constraint):
    from iris.fileformats.um import structured_um_loading
    with structured_um_loading():
        cube = iris.load_cube(filepath,constraint)
    return cube

um_cdnc_cube = {}
um_ctop_cube = {}
for filen in ppfiles:
    utc0 = int(filen[-3:])
    p_cube = load_with(filen,iris.AttributeConstraint(STASH='m01s00i408'))
    theta_cube = load_with(filen,iris.AttributeConstraint(STASH='m01s00i004'))
    [air_density, temperature] = get_air_density(p_cube, theta_cube)

    cdnc_cube = load_with(filen,iris.AttributeConstraint(STASH='m01s00i075'))
    cdnc = cdnc_cube*air_density/1.e6 #cm-3

    lwc = load_with(filen.replace('pj','pf'),iris.AttributeConstraint(STASH='m01s00i254'))
    cf  = load_with(filen.replace('pj','pf'),iris.AttributeConstraint(STASH='m01s00i267'))
    cf2 = cf
    cf2.data[np.where(cf.data<0.05)] = 0.05
    lwc = lwc/cf
    lwc2 = lwc
    lwc2.data[np.where(lwc.data<1.e-5)]=0.

    cdnc.data[np.where(lwc.data<1.e-5)]=np.nan
    cdnc.data[np.where(cf.data<0.05)]=np.nan

    umctop=np.zeros([2,53,len(cf.data[0,0,:,0]),len(cf.data[0,0,0,:])],dtype=float)
    umctop[:] = -999
    um_cdncs = load_with(path_pafile,cube3d)
    um_cdncs.data[:] = -999
    for it in range(2):
        print('i time',it)
        for ih in range(53):
            if ih > 0:
               iih = ih-1                                                     # choose 1 level below the actual cloud top height
            elif ih == 0:
               iih = 0
            umctop[it,ih,:,:] = np.where(lwc2.data[it,ih,:,:]>0,hr[iih],-999) # get the heights where cloud liquid is positive

        umctop_locs = umctop[it].argmax(axis=0)                               # get the maximum heights of + cloud liquid (clout top)
        a1,a2 = np.indices(umctop_locs.shape)                              
        um_cdncs.data[it] = cdnc.data[it,umctop_locs,a1,a2]                   # store the cdnc at cloud top heights

    for t in range(2):
        if int(daystring)>int(first_day_of_the_second_cycle):
            daystring2 = str(int(daystring)-cycle_len)
        else:
            daystring2 = daystring
        utc=utc0+t*3-(int(daystring2)-int(first_day_of_cycle))*24

        cdnc_r = um_cdncs[t,:,:]

        #### generate Iris cube of MODIS data, also target to regrid model to, from defined target coordinate system    
        modis_cube = iris.cube.Cube(CDNC_per_hour[str(utc)], var_name='target', dim_coords_and_dims=[
            (iris.coords.DimCoord(lat2[:,0],'latitude',units='degrees',coord_system=GeogCS(6371229)),0),
            (iris.coords.DimCoord(lon2[0,:],'longitude',units='degrees',coord_system=GeogCS(6371229)),1)])

        cdnc_s = cdnc_r.regrid(modis_cube, iris.analysis.Nearest(extrapolation_mode='mask'))
        print(cdnc_s.shape)
        print('regridding finished')

        if utc>=10:
            np.savetxt(path_to_save_cdnc_per_hour+'cdnc_at_ctop_'+daystring+'th_'+str(utc)+'utc.txt', cdnc_s.data, fmt='%.18e', delimiter=',')
        else:
            np.savetxt(path_to_save_cdnc_per_hour+'cdnc_at_ctop_'+daystring+'th_0'+str(utc)+'utc.txt', cdnc_s.data, fmt='%.18e', delimiter=',')


# Now start to plot
def filter_data(modis,um):
    var1 = np.where(modis>0,um,np.nan)
    return var1

# Select the hours around noon local time for the MODIS and UM
fsmodis_cdnc = [path_to_save_cdnc_per_hour+'modis_cdnc_'+str(int(daystring)-1)+'th_20utc.txt',path_to_save_cdnc_per_hour+'modis_cdnc_'+str(int(daystring)-1)+'th_21utc.txt',path_to_save_cdnc_per_hour+'modis_cdnc_'+str(int(daystring)-1)+'th_22utc.txt',path_to_save_cdnc_per_hour+'modis_cdnc_'+str(int(daystring)-1)+'th_23utc.txt',path_to_save_cdnc_per_hour+'modis_cdnc_'+daystring+'th_00utc.txt',path_to_save_cdnc_per_hour+'modis_cdnc_'+daystring+'th_01utc.txt',path_to_save_cdnc_per_hour+'modis_cdnc_'+daystring+'th_02utc.txt']
ums_cdnc = [path_to_save_cdnc_per_hour+'cdnc_at_ctop_'+str(int(daystring)-1)+'th_18utc.txt',path_to_save_cdnc_per_hour+'cdnc_at_ctop_'+str(int(daystring)-1)+'th_21utc.txt',path_to_save_cdnc_per_hour+'cdnc_at_ctop_'+str(int(daystring)-1)+'th_21utc.txt',path_to_save_cdnc_per_hour+'cdnc_at_ctop_'+str(int(daystring)-1)+'th_21utc.txt',path_to_save_cdnc_per_hour+'cdnc_at_ctop_'+daystring+'th_00utc.txt',path_to_save_cdnc_per_hour+'cdnc_at_ctop_'+daystring+'th_00utc.txt',path_to_save_cdnc_per_hour+'cdnc_at_ctop_'+daystring+'th_00utc.txt']

mcdnc = []
umcdnc = []
for fmodis_cdnc,um_cdnc,ij in zip(fsmodis_cdnc,ums_cdnc,np.arange(len(fsmodis_cdnc))):
    fmodis_cdnc0 = loadtxt(fmodis_cdnc,delimiter=',')
    um_cdnc0     = loadtxt(um_cdnc,delimiter=',')

    # Replace the data to np.nan where MODIS CDNC <= 0
    fmodis_cdnc = filter_data(fmodis_cdnc0,fmodis_cdnc0)
    um_cdnc     = filter_data(fmodis_cdnc0,um_cdnc0)

    # Save the data that do not overlap among the selected hours
    if ij == 0:
        pmcdnc = fmodis_cdnc
        pumcdnc = um_cdnc
    else:
        pmcdnc = np.where(~np.isnan(pmcdnc),pmcdnc,fmodis_cdnc)
        pumcdnc = np.where(~np.isnan(pumcdnc),pumcdnc,um_cdnc)

fig1,ax1 = plt.subplots(1,subplot_kw=dict(projection=ccrs.PlateCarree()))
ax1.set_extent([110, 180, -65, -30],crs=ccrs.PlateCarree())
im1=ax1.pcolormesh(lon,lat,pmcdnc,vmin=0,vmax=100)
plt.gca().coastlines()
ax1.gridlines()
plt.xticks(np.arange(lon[0,:].min(),lon[0,:].max(),7))
plt.yticks(np.arange(lat[:,0].min(),lat[:,0].max(),7))
bar = plt.colorbar(im1, orientation='horizontal', ax=ax1)
bar.ax.set_xlabel('MODIS CDNC / cm$^{-3}$',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax1.set_title('MODIS CDNC '+daystring+' March 2018')
fig1.savefig(daystring+'th_modis_map_cdnc.jpg')

fig2,ax2 = plt.subplots(1,subplot_kw=dict(projection=ccrs.PlateCarree()))
ax2.set_extent([110, 180, -65, -30],crs=ccrs.PlateCarree())
im2=plt.pcolormesh(lon,lat,pumcdnc,vmin=0,vmax=100)
plt.gca().coastlines()
ax2.gridlines()
plt.xticks(np.arange(lon[0,:].min(),lon[0,:].max(),7))
plt.yticks(np.arange(lat[:,0].min(),lat[:,0].max(),7))
bar = plt.colorbar(im2, orientation='horizontal', ax=ax2) 
bar.ax.set_xlabel('UM CDNC / cm$^{-3}$',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax2.set_title('CDNC '+daystring+' March 2018')
fig2.savefig(daystring+'th_um_map_cdnc.jpg')

x, y = pmcdnc.reshape(-1), pumcdnc.reshape(-1)
x = pmcdnc.reshape(-1)[np.isfinite(pumcdnc.reshape(-1))]
y = pumcdnc.reshape(-1)[np.isfinite(pumcdnc.reshape(-1))]

fig,ax2d = plt.subplots(1)
im=ax2d.hist2d(x,y,bins=50,vmax=30000,vmin=1,norm=mpl.colors.LogNorm())
if len(x)!= 0 and len(y)!=0:
    m, b = np.polyfit(x,y, 1)
    print('m,b',m,b)
    ax2d.plot(x,m*x,color='red',label='y={:.4f}x'.format(m)+'+{:.4f}'.format(b))

ax2d.set_xlabel('MODIS CDNC')
ax2d.set_ylabel('UM CDNC at cloud top')
ax2d.set_title('2D histogram CDNC')
ax2d.legend(frameon=False)
fig.colorbar(im[3], ax=ax2d)
fig.savefig(daystring+'th_2Dhist_cdnc_xlim.jpg')
