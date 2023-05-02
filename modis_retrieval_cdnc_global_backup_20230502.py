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
daystring = str(day)
modis_l2_init='MOD06_L2.A'


path_model1='/ocean/projects/atm200005p/xwangw/netcdfs/u-ct879/march/regional/201803'

#define and check target grid - the grid we will present results on from both satellite and model
# the model runs with 280x360 lat,lon grid cells, but our domain is larger. For now, keep grid cells approximately the same size

area_def = geometry.AreaDefinition.from_extent(area_id='southern_ocean',projection=ccrs.PlateCarree(),shape=(7722,2882),area_extent=(109.7,-61.4,179.9,-35.2),units='deg') 

lon,lat = area_def.get_lonlats()
#we rely on these 2D coordinates actually being copies of 1D coordinates, so check they're the same through the array
print('check lon',lon.shape)
print('check lat',lat.shape)

#####Need to assemble lists of MODIS 'granules' in swaths in domain. One 'granule' is one file. The next two subroutines below do this.
#let's assume we will never want more than 3 MODIS swaths and assume we never want more than three files in the same swath
#files in the same swath are separated by 5 minutes
#swaths are separated by 90 minutes
# The code depends on all of these things being true!


#######start to calculate CDNC from MODIS#######
#The regridded results are a list of up to three numpy arrays, each covering the whole domain, but each masked and only corresponding to one swath
#Choose unmasked data points from each to combine into a single numpy array, then turn that into an iris cube that will be used as the target

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
    nd_prefactor = (2*math.sqrt(10.0)/(k*math.pi*Q**3.0))
    effradius = df['Cloud_Effective_Radius_37']
    cot = df['Cloud_Optical_Thickness_37']
    lwp = df['Cloud_Water_Path_37']
    phase=df['Cloud_Phase_Infrared']
    temperature = df['Cloud_Top_Temperature']
    sza = df['Solar_Zenith']

    latitude = df['latitude']
    longitude = df['longitude']

    effdata = effradius
    cotdata = cot
    tempdata = temperature
    Cw = dqdz(tempdata,p)

    print('nd_prefactor = '+str(nd_prefactor),'Cw',Cw)

    nd_sqrtarg_num = f*Cw*cotdata
    nd_sqrtarg_den = rho_w*np.power(1e-6*effdata,5.0) #Grosvenor & Wood; effradius is in microns
    nd1 = 1e-6*nd_prefactor*np.sqrt(np.divide(nd_sqrtarg_num,nd_sqrtarg_den)) # units cm^-3
    #Nd1[np.isnan(Nd1)]=0.0
    #Nd1[np.isinf(Nd1)]=0.0
#    Nd2 = ma.masked_where(tempdata<273,Nd1)
    nd1 = np.where(~np.isnan(nd1),nd1,0)
    nd1 = np.where(~np.isinf(nd1),nd1,0)
    nd2 = ma.masked_where(phase>=2,nd1) # remove ice clouds, 0 -- cloud free, 1 -- water cloud, 2 -- ice cloud, 3 -- mixed phase cloud, 6 -- undetermined phase
    nd2 = ma.masked_where(phase<1,nd2) # remove ice clouds, 0 -- cloud free, 1 -- water cloud, 2 -- ice cloud, 3 -- mixed phase cloud, 6 -- undetermined phase
    nd3 = ma.masked_where(lwp<20,nd2)    # remove thin clouds
    nd4 = ma.masked_where(cot<3,nd3)    # remove thin clouds
    nd  = ma.masked_where(sza<55,nd4)   # get rid of the grids with solar zenith angles that are too low
    return nd,latitude,longitude

CDNC_per_file = {}
lat_rg = {}
lon_rg = {}
allfiles = sorted(glob.glob(modis_path+modis_l2_init+year+str(doy)+'*.nc'))[:46]
for ij,fn in enumerate(allfiles):
    df = xr.open_dataset(fn)
    Nd,lats0,lons0 = getCDNC(df)
    CDNC_per_file[str(ij)] = Nd
    lat_rg[str(ij)] = lats0
    lon_rg[str(ij)] = lons0
    print('ij',ij,'cdnc shape xxx',Nd.shape)

print('CDNC_per_file.keys=',CDNC_per_file.keys())

if daystring == '18':
    number_of_files = {'0':10,'1':3,'2':8,'3':7,'4':4,'5':11,'6':3}#,'7':8,'8':8,'9':3,'10':11,'11':4,'12':7,'13':8,'14':3,'15':10,'16':5,'17':6,'18':9,'19':3,'20':10,'21':5,'22':6,'23':10}
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
def regrid_modis_data(cdnc,lat_rg,lon_rg,number_of_files,regrid_method):
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

CDNC_per_hour = regrid_modis_data(CDNC_per_file,lat_rg,lon_rg,number_of_files,1)


for ij in range(24):
    print(CDNC_per_hour[str(ij)])
    if len(CDNC_per_hour[str(ij)])==0:
        print('plotting skip')
        continue

    CDNC_plot = np.where(CDNC_per_hour[str(ij)]>0,CDNC_per_hour[str(ij)],np.nan)
    CDNC_plot = np.where(CDNC_plot<5000,CDNC_plot,np.nan)

    #####Plot the combined, regridded MODIS data
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
    plt.pcolormesh(lon[0,:],lat[:,0],CDNC_plot,vmin=0,vmax=200)#, norm=LogNorm(vmin=0.03,vmax=3))
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
sys.exit()

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
        utc=utc0+t*3-(int(daystring)-18)*24
        for levr,height in zip(levrs,heights):
            print('height=',height)
            cdnc_s = cdnc[t,levr,:,:]

            #swath_def = geometry.SwathDefinition(lons=lon,lats=lat)
            #result = kd_tree.resample_nearest(swath_def, cdnc_s, area_def, radius_of_influence=1000, epsilon=0.5, fill_value=None)
            #result = np.ma.masked_where(result<=0,result)
            
            #### generate Iris cube of MODIS data, also target to regrid model to, from defined target coordinate system    
            modis_cube = iris.cube.Cube(CDNC_per_hour[str(utc)], var_name='target', dim_coords_and_dims=[
                (iris.coords.DimCoord(lat[:,0],'latitude',units='degrees',coord_system=GeogCS(6371229)),0),
                (iris.coords.DimCoord(lon[0,:],'longitude',units='degrees',coord_system=GeogCS(6371229)),1)])
            print('modis_cube',modis_cube)

            result2 = cdnc_s.regrid(modis_cube, iris.analysis.Linear(extrapolation_mode='mask'))
            print('results2.shape',result2.shape)
            print(result2)

            if height == '100m':
                um_cdnc_cube100m[str(utc)]=result2
            elif height == '500m':
                um_cdnc_cube500m[str(utc)]=result2
            elif height == '1km':
                um_cdnc_cube1km[str(utc)]=result2
            elif height == '2km':
                um_cdnc_cube2km[str(utc)]=result2
            elif height == '3km':
                um_cdnc_cube3km[str(utc)]=result2

                        
            #### Plot the regridded simulation data
            #fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
            #qplt.pcolormesh(result, norm=LogNorm(vmin=0.03,vmax=3))
            #plt.gca().coastlines()
            #ax.gridlines()
            #fig.savefig(str(time)+height+'_kd_tree_cdnc.jpg')

            '''
            #### Plot the regridded simulation data
            fig, ax = plt.subplots(1,subplot_kw=dict(projection=ccrs.PlateCarree()))
            ax.set_extent([110, 180, -65, -30],crs=ccrs.PlateCarree())
            plt.pcolormesh(lon,lat,result2.data,vmin=0,vmax=200)#, norm=LogNorm(vmin=0.03,vmax=3))
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
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)
    new_cube = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
    return new_cube

avg100m = []
avg500m = []
avg1km = []
avg2km = []
avg3km = []

avgmodis = []

for key in um_cdnc_cube100m.keys():
    print('UM UTC=',key)
    for modis_key in range(3):
        modis_key = str(int(key)+modis_key)
        print('MODIS UTC=',modis_key)
        modis_nd = CDNC_per_hour[modis_key]

        print('MODIS Nd',modis_nd.shape)
        if len(modis_nd[modis_nd>0]):
            um_nd0 = um_cdnc_cube100m[key]
            um_nd100m = um_nd0
            #um_nd1 = ma.masked_where(modis_nd>0,um_nd0.data)
            #um_nd100m.data = um_nd0.data[um_nd1.mask]
            um_nd100m.data = np.where(modis_nd>0,um_nd0.data,np.nan)
            um_nd100m.data = np.where(modis_nd<5000,um_nd100m.data,np.nan)
  
            um_nd0 = um_cdnc_cube500m[key]
            um_nd500m = um_nd0
            #um_nd1 = ma.masked_where(modis_nd>0,um_nd0.data)
            #um_nd500m.data = um_nd0.data[um_nd1.mask]
            um_nd500m.data = np.where(modis_nd>0,um_nd0.data,np.nan)
            um_nd500m.data = np.where(modis_nd<5000,um_nd500m.data,np.nan)
  
            um_nd0 = um_cdnc_cube1km[key]
            um_nd1km = um_nd0
            #um_nd1 = ma.masked_where(modis_nd>0,um_nd0.data)
            #um_nd1km.data = um_nd0.data[um_nd1.mask]
            um_nd1km.data = np.where(modis_nd>0,um_nd0.data,np.nan)
            um_nd1km.data = np.where(modis_nd<5000,um_nd1km.data,np.nan)

            um_nd0 = um_cdnc_cube2km[key]
            um_nd2km = um_nd0
            #um_nd1 = ma.masked_where(modis_nd>0,um_nd0.data)
            #um_nd2km.data = um_nd0.data[um_nd1.mask]
            um_nd2km.data = np.where(modis_nd>0,um_nd0.data,np.nan)
            um_nd2km.data = np.where(modis_nd<5000,um_nd2km.data,np.nan)
 
            um_nd0 = um_cdnc_cube3km[key]
            um_nd3km = um_nd0
            #um_nd1 = ma.masked_where(modis_nd>0,um_nd0.data)
            #um_nd3km.data = um_nd0.data[um_nd1.mask]
            um_nd3km.data = np.where(modis_nd>0,um_nd0.data,np.nan)
            um_nd3km.data = np.where(modis_nd<5000,um_nd3km.data,np.nan)

            modis_nd = np.where(modis_nd>0,modis_nd,np.nan)
            modis_nd = np.where(modis_nd<5000,modis_nd,np.nan)
 
            #avg100m.append(weighted_avg(um_nd100m))
            #avg500m.append(weighted_avg(um_nd500m))
            #avg1km.append(weighted_avg(um_nd1km))
            #avg2km.append(weighted_avg(um_nd2km))
            #avg3km.append(weighted_avg(um_nd3km))
  
            #avgmodis.append(weighted_avg(modis_nd))
       
            avg100m.append(np.nanmean(um_nd100m))
            avg500m.append(np.nanmean(um_nd500m))
            avg1km.append(np.nanmean(um_nd1km))
            avg2km.append(np.nanmean(um_nd2km))
            avg3km.append(np.nanmean(um_nd3km))
 
            avgmodis.append(np.nanmean(modis_nd))
            print('modis max',np.nanmax(modis_nd))
 
        else:
            print('skip scatter')
            continue
 
        fig,ax1 = plt.subplots(1)
        ax1.scatter(modis_nd,um_nd100m.data)
        x, y = modis_nd.reshape(-1), um_nd100m.data.reshape(-1)
        print('x.shape=',x.shape,'y.shape=',y.shape)
        print('isnan:','x.shape=',x[np.isfinite(y)].shape,'y.shape=',y[np.isfinite(y)].shape)

        m, b = np.polyfit(x[np.isfinite(y)],y[np.isfinite(y)], 1)
        print('100m m,b',m,b)
        ax1.plot(x[np.isfinite(y)],m*x[np.isfinite(y)]+b,color='black',label='y={:.4f}x'.format(m)+'+{:.4f}'.format(b))

        ax1.set_xlabel('MODIS CDNC')
        ax1.set_ylabel('UM CDNC at 100 m altitude')
        ax1.set_title(modis_key+' UTC '+daystring+' March 2018')
        ax1.legend(frameon=False)
        if int(modis_key)<10:
            fig.savefig(daystring+'_0'+modis_key+'_modis_um_scatter_cdnc_100m.jpg')
        else:
            fig.savefig(daystring+'_'+modis_key+'_modis_um_scatter_cdnc_100m.jpg')
         
        fig,ax1 = plt.subplots(1)
        ax1.scatter(modis_nd,um_nd500m.data)
        x, y = modis_nd.reshape(-1), um_nd500m.data.reshape(-1)

        m, b = np.polyfit(x[np.isfinite(y)],y[np.isfinite(y)], 1)
        print('500m m,b',m,b)
        ax1.plot(x[np.isfinite(y)],m*x[np.isfinite(y)]+b,color='black',label='y={:.4f}x'.format(m)+'+{:.4f}'.format(b))

        ax1.set_xlabel('MODIS CDNC')
        ax1.set_ylabel('UM CDNC at 500 m altitude')
        ax1.set_title(modis_key+' UTC '+daystring+' March 2018')
        ax1.legend(frameon=False)
        if int(modis_key)<10:
            fig.savefig(daystring+'_0'+modis_key+'_modis_um_scatter_cdnc_500m.jpg')
        else:
            fig.savefig(daystring+'_'+modis_key+'_modis_um_scatter_cdnc_500m.jpg')
 
        fig,ax1 = plt.subplots(1)
        ax1.scatter(modis_nd,um_nd1km.data)
        x, y = modis_nd.reshape(-1), um_nd1km.data.reshape(-1)

        m, b = np.polyfit(x[np.isfinite(y)],y[np.isfinite(y)], 1)
        print('1km m,b',m,b)
        ax1.plot(x[np.isfinite(y)],m*x[np.isfinite(y)]+b,color='black',label='y={:.4f}x'.format(m)+'+{:.4f}'.format(b))
        ax1.set_xlabel('MODIS CDNC')
        ax1.set_ylabel('UM CDNC at 1 km altitude')
        ax1.set_title(modis_key+' UTC '+daystring+' March 2018')
        ax1.legend(frameon=False)
        if int(modis_key)<10:
            fig.savefig(daystring+'_0'+modis_key+'_modis_um_scatter_cdnc_1km.jpg')
        else:
            fig.savefig(daystring+'_'+modis_key+'_modis_um_scatter_cdnc_1km.jpg')
 
        fig,ax1 = plt.subplots(1)
        ax1.scatter(modis_nd,um_nd2km.data)
        x, y = modis_nd.reshape(-1), um_nd2km.data.reshape(-1)

        m, b = np.polyfit(x[np.isfinite(y)],y[np.isfinite(y)], 1)
        print('2km m,b',m,b)
        ax1.plot(x[np.isfinite(y)],m*x[np.isfinite(y)]+b,color='black',label='y={:.4f}x'.format(m)+'+{:.4f}'.format(b))
        ax1.set_xlabel('MODIS CDNC')
        ax1.set_ylabel('UM CDNC at 2 km altitude')
        ax1.set_title(modis_key+' UTC '+daystring+' March 2018')
        ax1.legend(frameon=False)
        if int(modis_key)<10:
            fig.savefig(daystring+'_0'+modis_key+'_modis_um_scatter_cdnc_2km.jpg')
        else:
            fig.savefig(daystring+'_'+modis_key+'_modis_um_scatter_cdnc_2km.jpg')
 
        fig,ax1 = plt.subplots(1)
        ax1.scatter(modis_nd,um_nd3km.data)
        x, y = modis_nd.reshape(-1), um_nd3km.data.reshape(-1)

        m, b = np.polyfit(x[np.isfinite(y)],y[np.isfinite(y)], 1)
        print('3km m,b',m,b)
        ax1.plot(x[np.isfinite(y)],m*x[np.isfinite(y)]+b,color='black',label='y={:.4f}x'.format(m)+'+{:.4f}'.format(b))
        ax1.set_xlabel('MODIS CDNC')
        ax1.set_ylabel('UM CDNC at 3 km altitude')
        ax1.set_title(modis_key+' UTC '+daystring+' March 2018')
        ax1.legend(frameon=False)
        if int(modis_key)<10:
            fig.savefig(daystring+'_0'+modis_key+'_modis_um_scatter_cdnc_3km.jpg')
        else:
            fig.savefig(daystring+'_'+modis_key+'_modis_um_scatter_cdnc_3km.jpg')

        #### Plot the regridded simulation data
        fig, ax = plt.subplots(1,subplot_kw=dict(projection=ccrs.PlateCarree()))
        ax.set_extent([110, 180, -65, -30],crs=ccrs.PlateCarree())
        plt.pcolormesh(lon,lat,um_nd100m.data,vmin=0,vmax=200)#, norm=LogNorm(vmin=0.03,vmax=3))
        plt.gca().coastlines()
        ax.gridlines()
        plt.xticks(np.arange(lon[0,:].min(),lon[0,:].max(),7))
        plt.yticks(np.arange(lat[:,0].min(),lat[:,0].max(),7))
        cbar = plt.colorbar(orientation='horizontal')
        ax.set_title(str(modis_key)+' UTC '+daystring+' March 2018')
        if int(modis_key)<10:
            fig.savefig(daystring+'_0'+str(modis_key)+'_map_cdnc_100m.jpg')
        else:
            fig.savefig(daystring+'_'+str(modis_key)+'_map_cdnc_100m.jpg')

        fig, ax = plt.subplots(1,subplot_kw=dict(projection=ccrs.PlateCarree()))
        ax.set_extent([110, 180, -65, -30],crs=ccrs.PlateCarree())
        plt.pcolormesh(lon,lat,um_nd500m.data,vmin=0,vmax=200)#, norm=LogNorm(vmin=0.03,vmax=3))
        plt.gca().coastlines()
        ax.gridlines()
        plt.xticks(np.arange(lon[0,:].min(),lon[0,:].max(),7))
        plt.yticks(np.arange(lat[:,0].min(),lat[:,0].max(),7))
        cbar = plt.colorbar(orientation='horizontal')
        ax.set_title(str(modis_key)+' UTC '+daystring+' March 2018')
        if int(modis_key)<10:
            fig.savefig(daystring+'_0'+str(modis_key)+'_map_cdnc_500m.jpg')
        else:
            fig.savefig(daystring+'_'+str(modis_key)+'_map_cdnc_500m.jpg')

        fig, ax = plt.subplots(1,subplot_kw=dict(projection=ccrs.PlateCarree()))
        ax.set_extent([110, 180, -65, -30],crs=ccrs.PlateCarree())
        plt.pcolormesh(lon,lat,um_nd1km.data,vmin=0,vmax=200)#, norm=LogNorm(vmin=0.03,vmax=3))
        plt.gca().coastlines()
        ax.gridlines()
        plt.xticks(np.arange(lon[0,:].min(),lon[0,:].max(),7))
        plt.yticks(np.arange(lat[:,0].min(),lat[:,0].max(),7))
        cbar = plt.colorbar(orientation='horizontal')
        ax.set_title(str(modis_key)+' UTC '+daystring+' March 2018')
        if int(modis_key)<10:
            fig.savefig(daystring+'_0'+str(modis_key)+'_map_cdnc_1km.jpg')
        else:
            fig.savefig(daystring+'_'+str(modis_key)+'_map_cdnc_1km.jpg')

        fig, ax = plt.subplots(1,subplot_kw=dict(projection=ccrs.PlateCarree()))
        ax.set_extent([110, 180, -65, -30],crs=ccrs.PlateCarree())
        plt.pcolormesh(lon,lat,um_nd2km.data,vmin=0,vmax=200)#, norm=LogNorm(vmin=0.03,vmax=3))
        plt.gca().coastlines()
        ax.gridlines()
        plt.xticks(np.arange(lon[0,:].min(),lon[0,:].max(),7))
        plt.yticks(np.arange(lat[:,0].min(),lat[:,0].max(),7))
        cbar = plt.colorbar(orientation='horizontal')
        ax.set_title(str(modis_key)+' UTC '+daystring+' March 2018')
        if int(modis_key)<10:
            fig.savefig(daystring+'_0'+str(modis_key)+'_map_cdnc_2km.jpg')
        else:
            fig.savefig(daystring+'_'+str(modis_key)+'_map_cdnc_2km.jpg')

        fig, ax = plt.subplots(1,subplot_kw=dict(projection=ccrs.PlateCarree()))
        ax.set_extent([110, 180, -65, -30],crs=ccrs.PlateCarree())
        plt.pcolormesh(lon,lat,um_nd3km.data,vmin=0,vmax=200)#, norm=LogNorm(vmin=0.03,vmax=3))
        plt.gca().coastlines()
        ax.gridlines()
        plt.xticks(np.arange(lon[0,:].min(),lon[0,:].max(),7))
        plt.yticks(np.arange(lat[:,0].min(),lat[:,0].max(),7))
        cbar = plt.colorbar(orientation='horizontal')
        ax.set_title(str(modis_key)+' UTC '+daystring+' March 2018')
        if int(modis_key)<10:
            fig.savefig(daystring+'_0'+str(modis_key)+'_map_cdnc_3km.jpg')
        else:
            fig.savefig(daystring+'_'+str(modis_key)+'_map_cdnc_3km.jpg')

fig, ax1 = plt.subplots(1)
ax1.plot(np.arange(len(avgmodis)),avgmodis,label='MODIS')
ax1.plot(np.arange(len(avg100m)),avg100m,label='UM 100m') 
ax1.plot(np.arange(len(avg500m)),avg500m,label='UM 500m') 
ax1.plot(np.arange(len(avg1km)),avg1km,label='UM 1km') 
ax1.plot(np.arange(len(avg2km)),avg2km,label='UM 2km')  
ax1.plot(np.arange(len(avg3km)),avg3km,label='UM 3km')  
ax1.set_xlabel('time point')
ax1.set_ylabel('Averaged CDNC')
ax1.legend(frameon=False)
fig.savefig('area_weighted_cdnc.jpg')



