# This module contains functions useful for writing data to file
# and loading data in from files 

from netCDF4 import Dataset
import numpy as np

# Writing to netCDF file
def Write_to_nc(map_X, map_Y, map_BED, map_B, map_B_std, map_C, map_C_std, filename):
    ncfile = Dataset(filename,mode='w',format='NETCDF4')
    ncfile.title = 'Output Bed Elevation'
    # Make data dimensions
    ncfile.createDimension('x', map_X.shape[1])     # x axis
    ncfile.createDimension('y', map_X.shape[0])     # y axis
    # Create the axes/variable lengths
    x = ncfile.createVariable('x', np.float32, ('x',))
    x.units = 'km (stereographic) EW'
    y = ncfile.createVariable('y', np.float32, ('y',))
    y.units = 'km (stereographic) NS'
    # Define 2D variables to hold the data
    bedmach = ncfile.createVariable('bedmach', np.float64,('y','x'))
    bedmach.units = 'm'
    bedmach.standard_name = 'Bedmachine bed'
    bed = ncfile.createVariable('bed',np.float64,('y','x'))
    bed.units = 'm'
    bed.standard_name = 'Inversion output bed'
    errbed = ncfile.createVariable('errbed',np.float64,('y','x'))
    errbed.units = 'm'
    errbed.standard_name = 'Standard deviation of bed elevation'
    slip = ncfile.createVariable('slip',np.float64,('y','x'))
    slip.units = ''
    slip.standard_name = 'Inversion output slipperiness'
    errslip = ncfile.createVariable('errslip',np.float64,('y','x'))
    errslip.units = 'm'
    errslip.standard_name = 'Standard deviation of bed slipperiness'
    # Write in the data
    x[:] = map_X[0,:]
    y[:] = map_Y[:,0]
    bedmach[:,:] = map_BED[:,:]  
    bed[:,:] = map_B[:,:].real
    errbed[:,:] = map_B_std[:,:].real
    slip[:,:] = map_C[:,:].real
    errslip[:,:] = map_C_std[:,:].real
    ncfile.close()