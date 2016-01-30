import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import numpy as np

n = np.int32(134217728)

start = drv.Event()
end = drv.Event()

x = np.random.normal(size = n)
x_short = np.random.normal(size = 8)

start.record()
dev_x = gpuarray.to_gpu(x)
dev_x_short = gpuarray.to_gpu(x_short)
end.record() 
end.synchronize()
print "Transfer to GPU time: %fs" %(start.time_till(end)*1e-3)


print "Timing vectorized exponentiation:"

start.record()
dev_expx_short = cumath.exp(dev_x_short)
end.record() 
end.synchronize()
print "GPU array calc time (initial): %fs" %(start.time_till(end)*1e-3)

start.record()
dev_expx = cumath.exp(dev_x)
end.record() 
end.synchronize()
print "GPU array calc time: %fs" %(start.time_till(end)*1e-3)

start.record()
exp_x = np.exp(x)
end.record() 
end.synchronize()
print "CPU calc time: %fs" %(start.time_till(end)*1e-3)

print "Timing vectorized dot product/sum of squares:"

start.record()
gpuarray.dot(dev_x_short,dev_x_short)
end.record() 
end.synchronize()
print "GPU array calc time (initial): %fs" %(start.time_till(end)*1e-3)

start.record()
gpuarray.dot(dev_x,dev_x)
end.record() 
end.synchronize()
print "GPU array calc time: %fs" %(start.time_till(end)*1e-3)

start.record()
np.dot(x, x)
end.record() 
end.synchronize()
print "CPU calc time: %fs" %(start.time_till(end)*1e-3)
