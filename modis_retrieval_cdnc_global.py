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

# Script to regrid MODIS level 2 cloud and aerosol data and UM regional simulation data to the same grid for plotting and/or evaluation

#Assume all MODIS level 2 data is stored in the same directory

#####first figure out how to approximately match time for simulations and satellite
modis_path='/ocean/projects/atm200005p/xwangw/analysis/observations/modis/mod06_global/'


#Satellite data is stored for day of year. I want this for 2016, which was a leap year
year = '2018'
doy_array = [0,31,59,90,120,151,181,212,243,273,304,334]
day= 18 #I want to plot on 12 May 2016
month=3

doy = '0'+str(doy_array[month-1]+day) if doy_array[month-1]+day < 100 else str(doy_array[month-1]+day)

#need these for the model
monthstring = '0'+str(month) if month < 10 else str(month)
daystring = '18'
modis_l2_init='MOD06_L2.A'


#for these simulations, 3-hourly AOD for all aerosol modes is stored in umnsaa_pa000, and we want timeindex 17 (51 hours into run) for 12 May    
timeindex =0 #UTC can be found from MODIS file name
#we want index 9 for 18 May as it's 1 day earlier in the simulation
prefix='umnsaa'
hour_for_run_file='000'
path_model1='/ocean/projects/atm200005p/xwangw/netcdfs/u-ct879/march/regional/201803'

#define and check target grid - the grid we will present results on from both satellite and model
# the model runs with 280x360 lat,lon grid cells, but our domain is larger. For now, keep grid cells approximately the same size

area_def = geometry.AreaDefinition.from_extent(area_id='southern_ocean',projection=ccrs.PlateCarree(),shape=(7722,2882),area_extent=(109.7,-61.4,179.9,-35.2),units='deg') 

lon,lat = area_def.get_lonlats()
#we rely on these 2D coordinates actually being copies of 1D coordinates, so check they're the same through the array
print(lon.shape)
print('check lon')
print(lon[0,::100])
print(lon[1,::100])
print(lon[100,::100])
print('check lat')
print(lat.shape)
print(lat[::100,0])
print(lat[::100,1])
print(lat[::100,100])

#####Need to assemble lists of MODIS 'granules' in swaths in domain. One 'granule' is one file. The next two subroutines below do this.
#let's assume we will never want more than 3 MODIS swaths and assume we never want more than three files in the same swath
#files in the same swath are separated by 5 minutes
#swaths are separated by 90 minutes
# The code depends on all of these things being true!


#######start to calculate CDNC from MODIS#######
#The regridded results are a list of up to three numpy arrays, each covering the whole domain, but each masked and only corresponding to one swath
#Choose unmasked data points from each to combine into a single numpy array, then turn that into an iris cube that will be used as the target
#for regridding the model data

##if swaths overlap, which they would near the poles, just use the last one. Suboptimal but OK for now.
##I am not sure why I need to mask again
##print('swath 1 unmasked:',modis_regrid_results[0].count())
##swath1 = np.ma.masked_where(modis_regrid_results[0]<=0,modis_regrid_results[0])
##if len(modis_regrid_results)>1:
##    print('swath 2 unmasked:',modis_regrid_results[1].count())
##    swath2 = np.ma.masked_where(modis_regrid_results[1]<=0,modis_regrid_results[1])
##    all_modis_data = np.ma.where(swath1.mask,swath2,swath1)
##    if len(modis_regrid_results)>2:
##        print('swath 3 unmasked:',modis_regrid_results[2].count())
##        swath3 = np.ma.masked_where(modis_regrid_results[2]<=0,modis_regrid_results[2])
##        all_modis_data = np.ma.where(all_modis_data.mask,swath3,all_modis_data)
##else:
##    all_modis_data = modis_regrid_results[0]

def qsatw(T, p):
    Tzero=273.16
    T0 = np.copy(T)
    T0.fill(273.16)
    #parentheses=line continuation in python
    try:
        log10esw = (10.79574*(1-np.divide(T0,T))-5.028*np.log10(np.divide(T,T0))
                    + 1.50475e-4*(1-np.power(10,(-8.2369*(np.divide(T,T0)-1))))
                    + 0.42873e-3*(np.power(10,(4.76955*(1-np.divide(T,T0))))-1) + 2.78614)
    except Exception as e:
        print(str(e))
        print('exception at T='+str(T))
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

# this should work on either the original or the regridded, merged data, but
# I can't be bothered to make it work for the case when you patch together
# several original netcdf files. lus, using the original files will distort the latitudes and longitudes,
# since here I will assume the swath downloaded from MODIS has square pixels aligned with the left-hand and top
# axes of the dataset
# So I recommend regridding before CDNC is calculated, otherwise plotting the CDNC will only give approximate results
# Filename should be an array like ncname if regridded data is to be used, otherwise a single string
def getCDNC(df):
    k = 0.8 # Brenguier et al 2011; Martin et al 1994 apparently suggests 0.67 and Painemal & Zuidema 0.88, all for MBL. Grosvenor & Wood (2014) uses 0.8
    f = 0.7 # Grosvenor & Wood 2014
    Q=2
    rho_w = 1000 #kg m^-3
    p = 850*100.0 #Pa     ???xxx what is this? 
    Nd_prefactor = (2*math.sqrt(10.0)/(k*math.pi*Q**3.0))
    effradius = df['Cloud_Effective_Radius_37']
    cot = df['Cloud_Optical_Thickness_37']
    lwp = df['Cloud_Water_Path_37']
    phase=df['Cloud_Phase_Infrared_1km']
    temperature = df['cloud_top_temperature_1km']
    sza = df['Solar_Zenith']

    latitude = df['latitude']
    longitude = df['longitude']

    effdata = effradius
    cotdata = cot
    tempdata = temperature
    Cw = dqdz(tempdata,p)

    print('Nd_prefactor = '+str(Nd_prefactor))
    print('Cw',Cw)
    Nd_sqrtarg_num = f*Cw*cotdata
    Nd_sqrtarg_den = rho_w*np.power(1e-6*effdata,5.0) #Grosvenor & Wood; effradius is in microns
    Nd1 = 1e-6*Nd_prefactor*np.sqrt(np.divide(Nd_sqrtarg_num,Nd_sqrtarg_den)) # units cm^-3
    #Nd1[np.isnan(Nd1)]=0.0
    #Nd1[np.isinf(Nd1)]=0.0
#    Nd2 = ma.masked_where(tempdata<273,Nd1)
    Nd1 = np.where(~np.isnan(Nd1),Nd1,0)
    Nd1 = np.where(~np.isinf(Nd1),Nd1,0)
    Nd2 = ma.masked_where(phase>=2,Nd1) # remove ice clouds
    Nd3 = ma.masked_where(lwp<2,Nd2)    # remove thin clouds
    Nd4 = ma.masked_where(cot<3,Nd3)    # remove thin clouds
    Nd  = ma.masked_where(sza<55,Nd4)   # get rid of the grids with solar zenith angles that are too low
    return Nd,latitude,longitude

CDNC = {}
lat_rg = {}
lon_rg = {}
allfiles = sorted(glob.glob(modis_path+modis_l2_init+year+str(doy)+'*.nc'))
for ij,fn in enumerate(allfiles):
    df = xr.open_dataset(fn)
    nd,lats0,lons0 = getCDNC(df)
    CDNC[str(ij)] = nd
    lat_rg[str(ij)] = lats0
    lon_rg[str(ij)] = lons0
    print('ij',ij)
    print('cdnc shape xxx',nd.shape)

print(CDNC.keys())
print(CDNC['0'],CDNC['1'],CDNC['2'],len(CDNC))

if daystring == '18':
    number_of_files = {'0':10,'1':3,'2':8,'3':7,'4':4,'5':11,'6':3,'7':8,'8':8,'9':3,'10':11,'11':4,'12':7,'13':8,'14':3,'15':10,'16':5,'17':6,'18':9,'19':3,'20':10,'21':5,'22':6,'23':10}
    ppfiles = [path_model1+'18/umnsaa_pj000',path_model1+'18/umnsaa_pj006',path_model1+'18/umnsaa_pj012',path_model1+'18/umnsaa_pj018']
elif daystring == '19':
    number_of_files = {'0':3,'1':9,'2':6,'3':5,'4':10,'5':4,'6':8,'7':7,'8':4,'9':11,'10':3,'11':8,'12':7,'13':3,'14':11,'15':4,'16':7,'17':8,'18':3,'19':11,'20':5,'21':6,'22':9,'23':3}
    ppfiles = [path_model1+'18/umnsaa_pj024',path_model1+'18/umnsaa_pj030',path_model1+'18/umnsaa_pj036',path_model1+'18/umnsaa_pj042']
elif daystring == '20':
    number_of_files = {'0':10,'1':5,'2':6,'3':9,'4':3,'5':9,'6':6,'7':5,'8':10,'9':3,'10':9,'11':7,'12':4,'13':11,'14':3,'15':8,'16':7,'17':4,'18':10,'19':4,'20':7,'21':8,'22':3,'23':11}
    ppfiles = [path_model1+'18/umnsaa_pj048',path_model1+'18/umnsaa_pj054',path_model1+'18/umnsaa_pj060',path_model1+'18/umnsaa_pj066']
elif daystring == '21':
    number_of_files = {'0':4,'1':7,'2':9,'3':3,'4':10,'5':5,'6':6,'7':9,'8':3,'9':9,'10':6,'11':5,'12':10,'13':3,'14':9,'15':6,'16':5,'17':11,'18':3,'19':8,'20':7,'21':4,'22':11,'23':3}
    ppfiles = [path_model1+'20/umnsaa_pj024',path_model1+'20/umnsaa_pj030',path_model1+'20/umnsaa_pj036',path_model1+'20/umnsaa_pj042']
elif daystring == '22':
    number_of_files = {'0':7,'1':8,'2':3,'3':11,'4':4,'5':7,'6':8,'7':4,'8':10,'9':5,'10':6,'11':9,'12':3,'13':10,'14':6,'15':5,'16':10,'17':3,'18':9,'19':6,'20':5,'21':10,'22':4,'23':8}
    ppfiles = [path_model1+'20/umnsaa_pj048',path_model1+'20/umnsaa_pj054',path_model1+'20/umnsaa_pj060',path_model1+'20/umnsaa_pj066']

#####This subroutine reads in the MODIS data and regrids it, for one satellite instrument (e.g. MODIS on TERRA or MODIS on AQUA) and one day
def regrid_modis_data(CDNC,lat_rg,lon_rg,number_of_files,regrid_method):
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
                nd_temp = CDNC[str(acc_file+k)]
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
                #nearest-neighbour regrid is one of several options. This is sensible when source and target pixels are the same size. 
                #radius of influence is in meters, for each target pixel, will look up to (1+epsilon) x this many meters away for a source pixel - try 10km
                #fill_value=None will mask undetermined pixels
                #https://pyresample.readthedocs.io/en/latest/api/pyresample.html?highlight=concatenate#module-pyresample.kd_tree
                result = kd_tree.resample_nearest(big_swath_def, big_nd_array, area_def, radius_of_influence=1000, epsilon=0.5, fill_value=None)
            elif regrid_method==2:
                #Gaussian weighting will use several neighbors, so probably makes more sense if target grids are a little larger than source grids
                result = kd_tree.resample_gauss(big_swath_def, big_nd_array, area_def, radius_of_influence=10000, sigmas=5000, fill_value=None)
            result = np.ma.masked_where(result<=0,result)

            print('3rd shape',len(result))
        nd_plot[ij]=result

    return nd_plot

nd_plot = regrid_modis_data(CDNC,lat_rg,lon_rg,number_of_files,1)

'''
for ij in range(24):
    print(nd_plot[str(ij)])
    if len(nd_plot[str(ij)])==0:
        print('plotting skip')
        continue
    #####Plot the combined, regridded MODIS data
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
    plt.pcolormesh(lon[0,:],lat[:,0],nd_plot[str(ij)],vmin=0,vmax=200)#, norm=LogNorm(vmin=0.03,vmax=3))
    plt.gca().coastlines()
    plt.xticks(np.arange(lon[0,:].min(),lon[0,:].max(),7))
    plt.yticks(np.arange(lat[:,0].min(),lat[:,0].max(),7))

    ax.gridlines()
    cbar = plt.colorbar(orientation='horizontal')
    ax.set_title(str(ij)+' UTC '+daystring+' March 2018')
    if ij < 10:
        fig.savefig('cdnc_'+daystring+'_0'+str(ij)+'.jpg')
    else:
        fig.savefig('cdnc_'+daystring+'_'+str(ij)+'.jpg')
'''
def get_air_density( air_pressure,potential_temperature):
  p0 = iris.coords.AuxCoord(1000.0,
                            long_name='reference_pressure',
                            units='hPa')
  p0.convert_units(air_pressure.units)

  Rd=287.05 # J/kg/K
  cp=1005.46 # J/kg/K
  Rd_cp=Rd/cp

  temperature=potential_temperature*(air_pressure/p0)**(Rd_cp)
  #print temperature.data[0,0,0]
  temperature._var_name='temperature'
  R_specific=iris.coords.AuxCoord(287.058,
                                  long_name='R_specific',
                                  units='J-kilogram^-1-kelvin^-1')#J/(kgK)

  air_density=(air_pressure/(temperature*R_specific))
  air_density.long_name='Density of air'
  air_density._var_name='air_density'
  air_density.units='kg m-3'
  temperature.units='K'
  #print air_density.data[0,0,0]
  return [air_density, temperature]

def load_with(filepath,constraint):
    from iris.fileformats.um import structured_um_loading
    with structured_um_loading():
        cube = iris.load_cube(filepath,constraint)
    return cube

levrs = [4,11,15,23,28]
heights = ['100m','500m','1km','2km','3km']

um_cdnc_cube100m = {}
um_cdnc_cube500m = {}
um_cdnc_cube1km = {}
um_cdnc_cube2km = {}
um_cdnc_cube3km = {}
for filen in ppfiles:
    utc0 = int(filen[-3:])
    p_cube = load_with(filen,iris.AttributeConstraint(STASH='m01s00i408'))
    theta_cube = load_with(filen,iris.AttributeConstraint(STASH='m01s00i004'))
    [air_density, temperature] = get_air_density(p_cube, theta_cube)

    cdnc_cube = load_with(filen,iris.AttributeConstraint(STASH='m01s00i075'))
    cdnc = cdnc_cube*air_density/1.e6 #cm-3

    for t in range(2):
        utc=utc0+t*3
        for levr,height in zip(levrs,heights):
            print(height)
            cdnc_s = cdnc[t,levr,:,:]

            #swath_def = geometry.SwathDefinition(lons=lon,lats=lat)
            #result = kd_tree.resample_nearest(swath_def, cdnc_s, area_def, radius_of_influence=1000, epsilon=0.5, fill_value=None)
            #result = np.ma.masked_where(result<=0,result)
            
            #### generate Iris cube of MODIS data, also target to regrid model to, from defined target coordinate system    
            modis_cube = iris.cube.Cube(nd_plot[str(utc)], var_name='target', dim_coords_and_dims=[
                (iris.coords.DimCoord(lat[:,0],'latitude',units='degrees',coord_system=GeogCS(6371229)),0),
                (iris.coords.DimCoord(lon[0,:],'longitude',units='degrees',coord_system=GeogCS(6371229)),1)])
            print(modis_cube)

            result2 = cdnc_s.regrid(modis_cube, iris.analysis.Linear(extrapolation_mode='mask'))
            print(result2.shape)
            print(result2)

            if height == '100m':
                um_cdnc_cube100m[str(utc)]=cdnc[t,levr,:,:]
            elif height == '500m':
                um_cdnc_cube500m[str(utc)]=cdnc[t,levr,:,:]
            elif height == '1km':
                um_cdnc_cube1km[str(utc)]=cdnc[t,levr,:,:]
            elif height == '2km':
                um_cdnc_cube2km[str(utc)]=cdnc[t,levr,:,:]
            elif height == '3km':
                um_cdnc_cube3km[str(utc)]=cdnc[t,levr,:,:]

            '''
            #### Plot the regridded simulation data
            #fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
            #qplt.pcolormesh(result, norm=LogNorm(vmin=0.03,vmax=3))
            #plt.gca().coastlines()
            #ax.gridlines()
            #fig.savefig(str(time)+height+'_kd_tree_cdnc.jpg')

            #### Plot the regridded simulation data
            fig, ax = plt.subplots(1,subplot_kw=dict(projection=ccrs.PlateCarree()))
            ax.set_extent([110, 180, -65, -30],crs=ccrs.PlateCarree())
            plt.pcolormesh(lon,lat,result2.data)#, norm=LogNorm(vmin=0.03,vmax=3))
            plt.gca().coastlines()
            ax.gridlines()
            plt.xticks(np.arange(lon[0,:].min(),lon[0,:].max(),7))
            plt.yticks(np.arange(lat[:,0].min(),lat[:,0].max(),7))
            cbar = plt.colorbar(orientation='horizontal')
            ax.set_title(str(utc)+' UTC '+daystring+' March 2018')
            if utc<10:
                fig.savefig(daystring+'_0'+str(utc)+'_'+height+'_linear_cdnc.jpg')
            else:
                fig.savefig(daystring+'_'+str(utc)+'_'+height+'_linear_cdnc.jpg')
            '''

def weighted_avg(cube):
    cube.coord('grid_latitude').guess_bounds()
    cube.coord('grid_longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)
    new_cube = cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MEAN, weights=grid_areas)
    return new_cube

avg100m = []
avg500m = []
avg1km = []
avg2km = []
avg3km = []

avgmodis = []

for key in um_cdnc_cube100m.keys():
    print(key)
    modis_nd = nd_plot[key]

    if len(modis_nd[modis_nd>0]):
        print('skip scatter')
        continue

    um_nd0 = um_cdnc_cube100m[key]
    um_nd1 = ma.masked_where(modis_nd>0,um_nd0)
    um_nd100m = um_nd0[um_nd1.mask]

    um_nd0 = um_cdnc_cube500m[key]
    um_nd1 = ma.masked_where(modis_nd>0,um_nd0)
    um_nd500m = um_nd0[um_nd1.mask]

    um_nd0 = um_cdnc_cube1km[key]
    um_nd1 = ma.masked_where(modis_nd>0,um_nd0)
    um_nd1km = um_nd0[um_nd1.mask]

    um_nd0 = um_cdnc_cube2km[key]
    um_nd1 = ma.masked_where(modis_nd>0,um_nd0)
    um_nd2km = um_nd0[um_nd1.mask]

    um_nd0 = um_cdnc_cube3km[key]
    um_nd1 = ma.masked_where(modis_nd>0,um_nd0)
    um_nd3km = um_nd0[um_nd1.mask]

    avg100m.append(weighted_avg(um_nd100m))
    avg500m.append(weighted_avg(um_nd500m))
    avg1km.append(weighted_avg(um_nd1km))
    avg2km.append(weighted_avg(um_nd2km))
    avg3km.append(weighted_avg(um_nd3km))

    avgmodis.append(weighted_avg(modis_nd))
    
    fig,ax1 = plt.subplots(1)
    ax1.scatter(modis_nd,um_nd100m.data)
    ax1.set_xlabel('MODIS CDNC')
    ax1.set_ylabel('UM CDNC at 100 m altitude')
    ax1.set_title(key+' UTC '+daystring+' March 2018')
    if int(key)<10:
        fig.savefig(daystring+'_0'+key+'_'+height+'_modis_um_scatter_cdnc_100m.jpg')
    else:
        fig.savefig(daystring+'_'+key+'_'+height+'_modis_um_scatter_cdnc_100m.jpg')
   
    fig,ax1 = plt.subplots(1)
    ax1.scatter(modis_nd,um_nd500m.data)
    ax1.set_xlabel('MODIS CDNC')
    ax1.set_ylabel('UM CDNC at 500 m altitude')
    ax1.set_title(key+' UTC '+daystring+' March 2018')
    if int(key)<10:
        fig.savefig(daystring+'_0'+key+'_'+height+'_modis_um_scatter_cdnc_500m.jpg')
    else:
        fig.savefig(daystring+'_'+key+'_'+height+'_modis_um_scatter_cdnc_500m.jpg')

    fig,ax1 = plt.subplots(1)
    ax1.scatter(modis_nd,um_nd1km.data)
    ax1.set_xlabel('MODIS CDNC')
    ax1.set_ylabel('UM CDNC at 1 km altitude')
    ax1.set_title(key+' UTC '+daystring+' March 2018')
    if int(key)<10:
        fig.savefig(daystring+'_0'+key+'_'+height+'_modis_um_scatter_cdnc_1km.jpg')
    else:
        fig.savefig(daystring+'_'+key+'_'+height+'_modis_um_scatter_cdnc_1km.jpg')

    fig,ax1 = plt.subplots(1)
    ax1.scatter(modis_nd,um_nd2km.data)
    ax1.set_xlabel('MODIS CDNC')
    ax1.set_ylabel('UM CDNC at 2 km altitude')
    ax1.set_title(key+' UTC '+daystring+' March 2018')
    if int(key)<10:
        fig.savefig(daystring+'_0'+key+'_'+height+'_modis_um_scatter_cdnc_2km.jpg')
    else:
        fig.savefig(daystring+'_'+key+'_'+height+'_modis_um_scatter_cdnc_2km.jpg')

    fig,ax1 = plt.subplots(1)
    ax1.scatter(modis_nd,um_nd3km.data)
    ax1.set_xlabel('MODIS CDNC')
    ax1.set_ylabel('UM CDNC at 3 km altitude')
    ax1.set_title(key+' UTC '+daystring+' March 2018')
    if int(key)<10:
        fig.savefig(daystring+'_0'+key+'_'+height+'_modis_um_scatter_cdnc_3km.jpg')
    else:
        fig.savefig(daystring+'_'+key+'_'+height+'_modis_um_scatter_cdnc_3km.jpg')

fig, ax1 = plt.subplots(1)
ax1.plot(np.range(len(avgmodis)),avgmodis,label='MODIS')
ax1.plot(np.range(len(avg100m)),avg100m,label='UM 100m') 
ax1.plot(np.range(len(avg500m)),avg500m,label='UM 500m') 
ax1.plot(np.range(len(avg1km)),avg1km,label='UM 1km') 
ax1.plot(np.range(len(avg2km)),avg2km,label='UM 2km')  
ax1.plot(np.range(len(avg3km)),avg3km,label='UM 3km')  
ax1.set_xlabel('time point')
ax1.set_ylabel('Averaged CDNC')
ax1.legend(frameon=False)
fig.savefig('area_weighted_cdnc.jpg')
sys.exit()
#### Now do something similar for the simulations
# For AOD, we want to add contributions from aerosol modes and dust
def load_aod(path, prefix, tindex, dateindex):
    ait_aod = iris.load_cube(path+prefix+'_pa'+dateindex,iris.AttributeConstraint(STASH='m01s02i300'))[1,tindex,:,:]
    acc_aod = iris.load_cube(path+prefix+'_pa'+dateindex,iris.AttributeConstraint(STASH='m01s02i301'))[1,tindex,:,:]
    cor_aod = iris.load_cube(path+prefix+'_pa'+dateindex,iris.AttributeConstraint(STASH='m01s02i302'))[1,tindex,:,:]
    aiti_aod = iris.load_cube(path+prefix+'_pa'+dateindex,iris.AttributeConstraint(STASH='m01s02i303'))[1,tindex,:,:]
    total_aod = ait_aod+acc_aod+cor_aod+aiti_aod
    return total_aod

total_aod11 = load_aod(path_model1,prefix,timeindex,hour_for_run_file)
print(total_aod11)
print(total_aod11.coord('grid_longitude'))

#Regrid the simulation data
regridded_model_aod = total_aod11.regrid(modis_cube, iris.analysis.Linear(extrapolation_mode='mask'))

#### Plot the regridded simulation data
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
qplt.pcolormesh(regridded_model_aod, norm=LogNorm(vmin=0.03,vmax=3))
plt.gca().coastlines()
ax.gridlines()

def get_evaluation_metrics(modis_cube,model_cube):
    nmb = np.ma.sum(model_cube.data-modis_cube.data)/np.ma.sum(modis_cube.data)
    pearsonr = np.ma.corrcoef(model_cube.data.flatten(),modis_cube.data.flatten())
    print('NMB, R, ',nmb,pearsonr[0,1])

get_evaluation_metrics(modis_cube,regridded_model_aod)

plt.figure()
plt.scatter(modis_cube.data.flatten(),regridded_model_aod.data.flatten())
plt.xlabel('MODIS AOD')
plt.ylabel('Model AOD')
plt.show()

