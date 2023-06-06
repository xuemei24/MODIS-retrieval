1. Download MODIS Level2 data globally from https://ladsweb.modaps.eosdis.nasa.gov/search/order without any postprocessing.
2. Convert all the files to netCDF, so that the coordinates are in the same shape as the variables.
   a. Download the converting command from http://hdfeos.org/software/h4cflib.php. Save it to the server(s) and then run "./h4tonccf_nc4 **.hdf" on the command line.
3. Adapt the settings until line 73
