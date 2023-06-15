from iris.analysis.cartography import unrotate_pole
import iris
import iris.analysis.cartography as iac
import numpy as np

def unrotate_coord(path_coord,stashcodes,pole_lon,pole_lat):

    cube = iris.load_cube(path_coord,iris.AttributeConstraint(STASH=stashcodes))
    lat_read = cube.coord('grid_latitude').points
    lon_read = cube.coord('grid_longitude').points
    nX = len(lon_read)
    nY = len(lat_read)

    #Get the shape of the arrays (i.e. number of elements in array in each dimension
    #These arrays are single vectors of size N x 1
    sh_lat = lat_read.shape
    sh_lon = lon_read.shape

    #replicate the 1D arrays in 2D arrays using the shapes of the other array (e.g. shape of lat for the lon replication)
    lat2d=np.tile(lat_read,[sh_lon[0],1])
    lon2d=np.tile(lon_read,[sh_lat[0],1])
    lon2d_2=lon2d
    lat2d_2=np.transpose(lat2d)  #Transpose (swap dimensions) of the 2D lat array to make it correspond to the location on a 2D grid
    #Set x and y above depending on orientation and location of x-section

    #Unrotate the pole using the IRIS utility
    lon, lat = iac.unrotate_pole(lon2d_2,lat2d_2,pole_lon,pole_lat)
    print('unrotate pole done')

    return lon, lat
