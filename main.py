#
## general code
#
#you have the N 
#you have the W
#you have the gray value (which is the signal on the point cloud
#
#YOU NEED TO IMPLEMENT THE FORWARD EULER METHOD:
#	for each color channel
#
#as usual create the opencl context
#
#and do the data transfer 
#	(WAIT YOU NEED 1D 2D Buffer Objects or Image objects)
#
#and then wrote the kernel and run it
#
#	I would say create a 2d buffer object
#	NO for the moment let it be a gray scale object
#	
#	fNew = fOld(1-dt SUM) + dt(SUM fATv)
#
#
#
#Ask:
#	is weight differnet for every color component?
#	the formula for the weight might matter?

from time import time
import pyopencl as cl
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type = str, required=True,
    help="path to nbrs and wgts generated")
ap.add_argument("-t", "--texture", type = str, required=True,
    help="path to the texture, for the moment just for gray scale, npz format")
ap.add_argument("-k", "--k", type = int,  required=True,
    help="number of nns")
ap.add_argument("-it", "--iteration", type = int , required = True, help = "number of iterations") 
args = vars(ap.parse_args())

#load the available data
data = np.load(args["input"])
ngbrs = data['ngbrs']
wgts = data['weights']
ngbrs = ngbrs.astype('int32')
wgts = wgts.astype('float32')
gray = np.load(args["texture"])
gray = gray['texture']
gray = gray[0:]
gray = 255*gray
gray = gray.astype('float32')
k = args["k"]
n = len(gray)
dt =0.1
it = args["iteration"]
print("success till loading")

# CHECK THE DATA TYPES AGAIN
# create the opencl context
platform = cl.get_platforms()[0]
print(platform)

device = platform.get_devices()[0]
print(device)

context = cl.Context([device])
print(context)

program = cl.Program(context, open("kernel_lap.cl").read()).build()

queue = cl.CommandQueue(context)
print(queue)


#create the buffers now.
mem_flags = cl.mem_flags
ngbrs_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,hostbuf=ngbrs) 
intensity_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=gray)
weight_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=wgts)

#need to create new intensity buffers
new_gray = np.ndarray(shape=(n,), dtype=np.float32)
new_intensity_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, new_gray.nbytes)

#run the kernel here in a looop
# program.laplacian_filter(queue, (n,), None,intensity_buf, new_intensity_buf, ngbrs_buf, weight_buf, np.int32(k), np.float32(dt) )
# so there is a loop
program.laplacian_filter(queue, (n,), None,intensity_buf, new_intensity_buf, ngbrs_buf, weight_buf, np.int32(k), np.float32(dt) )
queue.finish()
for uv in range(0, it):
	program.laplacian_filter(queue, (n,), None,new_intensity_buf, new_intensity_buf, ngbrs_buf, weight_buf, np.int32(k), np.float32(dt) )
	queue.finish()
#	temp.kernel = program.laplacian_filter
#	temp.kernel.set_args(0,new_intensity_buf)
# copy the new intensity vec
queue.finish()
cl.enqueue_copy(queue, new_gray, new_intensity_buf)

# save the new intensity vec here
print("finish")
new_gray = new_gray/255
np.savez("./data/newGray.npz", texture=new_gray)
