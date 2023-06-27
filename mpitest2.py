from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

m = 1 # Sliding law constant
C = 100 # Mean slipperiness
# Data error levels
p_b = -2
p_c = -2
erB = 1 * (10 **(-3)) #0.000001#0000001
erC = 1 * (10 **(-3)) #0.000001#0000001

n = 3 # How many overlapping squares (n^2)
adj = [3,3]
square_size = 50000
tapering = 0.1 # Lose 10% on each side to tapering
centre_include = 1

# Create a list of central coordinates to run 

#centre_coord = [-1.41e6, -0.455e6] # Lower Thwaites
#centre_coord = [-1.30e6, -0.445e6] # Upper Thwaites (ish)
#centre_coord = [-1.586e6,-0.122e6] # PIG istar7 Hill
#centre_coord = [-1.586e6,-0.222e6] # PIG istar7 Hill
#centre_coord = [-1551047.869, -247996.3066] #istar 18

if rank == 0:
#    data = np.arange(15.0)
    centre_coords = ([-1.535e6,-0.125e6],\
                     [-1.535e6,-0.15e6],\
                     [-1.535e6,-0.175e6],\
                     [-1.535e6,-0.2e6],\
                     [-1.535e6,-0.225e6],\
                     [-1.535e6,-0.25e6],\
                     [-1.56e6,-0.125e6],\
                     [-1.56e6,-0.15e6],\
                     [-1.56e6,-0.175e6],\
                     [-1.56e6,-0.2e6],\
                     [-1.56e6,-0.225e6],\
                     [-1.56e6,-0.25e6],\
                     [-1.585e6,-0.125e6],\
                     [-1.585e6,-0.15e6],\
                     [-1.585e6,-0.175e6],\
                     [-1.585e6,-0.2e6],\
                     [-1.585e6,-0.225e6],\
                     [-1.585e6,-0.25e6],\
                     [-1.61e6,-0.125e6],\
                     [-1.61e6,-0.15e6],\
                     [-1.61e6,-0.175e6],\
                     [-1.61e6,-0.2e6],\
                     [-1.61e6,-0.225e6],\
                     [-1.61e6,-0.25e6],\
                     [-1.635e6,-0.125e6],\
                     [-1.635e6,-0.15e6],\
                     [-1.635e6,-0.175e6],\
                     [-1.635e6,-0.2e6],\
                     [-1.635e6,-0.225e6],\
                     [-1.635e6,-0.25e6])
 #   centre_coords = ([-1.61e6,-0.125e6],\
 #                    [-1.61e6,-0.15e6],\
 #                    [-1.61e6,-0.175e6],\
 #                    [-1.61e6,-0.2e6],\
 #                    [-1.61e6,-0.225e6],\
 #                    [-1.61e6,-0.25e6], \
 #                    [-1.585e6,-0.2e6],\
 #                    [-1.585e6,-0.225e6],\
 #                    [-1.585e6,-0.25e6],\
 #                    [-1.635e6,-0.175e6]    )
 #   centre_coords = ([-1.586e6, -0.123e6], \
 #                    [-1.6225e6, -0.143e6], \
 #                    [-1.56e6, -0.128e6], \
 #                    [-1.545e6, -0.183e6], \
 #                    [-1.543e6, -0.215e6], \
 #                    [-1.551e6, -0.248e6])
 #   centre_coords = ([-1.6225e6+5e3, -0.143e6], \
 #                    [-1.6225e6-5e3, -0.143e6], \
 #                    [-1.6225e6, -0.143e6+5e3], \
 #                    [-1.6225e6, -0.143e6-5e3])#, \
 #   centre_coords = ([-1.6225e6, -0.143e6], \
 #                    [-1.6225e6+10e3, -0.143e6], \
 #                    [-1.6225e6+15e3, -0.143e6], \
 #                    [-1.6225e6+10e3, -0.143e6+5e3], \
 #                    [-1.6225e6+10e3, -0.143e6-5e3], \
 #                    [-1.6225e6+15e3, -0.143e6+5e3], \
 #                    [-1.6225e6+15e3, -0.143e6-5e3])#, \
#                     [-1.551e6+5e3, -0.248e6], \
#                     [-1.551e6-5e3, -0.248e6], \
#                     [-1.551e6, -0.248e6+5e3], \
#                     [-1.551e6, -0.248e6-5e3])
    data = centre_coords


    # determine the size of each sub-task
    ave, res = divmod(len(data), nprocs)
    counts = [ave + 1 if p < res else ave for p in range(nprocs)]

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(nprocs)]
    ends = [sum(counts[:p+1]) for p in range(nprocs)]

    # converts data into a list of arrays 
    data = [data[starts[p]:ends[p]] for p in range(nprocs)]
else:
    data = None

data = comm.scatter(data, root=0)

#print('Process {} has data:'.format(rank), data)
#print(type(len(data)))

file_prefix = 'PIGv2_C100_2008_'
#file_prefix2 = 'bigPIG_C150_v2_'
#file_prefix3 = 'bigPIG_C200_v2_'

from inversion_module import terminal_inversion
#ter

#for rank in nprocs:
for i in range(len(data)):
#    print('Process {} version {} has data'.format(rank, i), data[i])
#    filename = file_prefix + str(rank) + '_' + str(i) + '.nc'
#    print(filename)
#    terminal_inversion(m, C, p_b, p_c, erB, erC, n, adj, square_size, tapering, centre_include, data[i], filename)
    filename = file_prefix + str(rank) + '_' + str(i) + '.nc'
    terminal_inversion(m, C, p_b, p_c, erB, erC, n, adj, square_size, tapering, centre_include, data[i], filename)
#    filename = file_prefix3 + str(rank) + '_' + str(i) + '.nc'
#    terminal_inversion(m, 200, p_b, p_c, erB, erC, n, adj, square_size, tapering, centre_include, data[i], filename)
    
#else:

#file_prefix = 'PIG_drot_REMA_29_16_'
#filename = file_prefix + str(rank) + '.nc'
#print(filename)

