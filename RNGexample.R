library(RCUDA)

cat("Setting cuGetContext(TRUE)...\n")
cuGetContext(TRUE)

ptx = nvcc("random.cu", out = "random.ptx", target = "ptx",
     "-arch=compute_20", "-code=sm_20,compute_20")
  
mod = loadModule(ptx)

setup = mod$setup_kernel
rnorm = mod$rnorm_kernel

n = as.integer(1e8)  # NOTE 'n' is of type integer
n_per_thread = as.integer(1000)

mu = 0.3
sigma = 1.5

verbose = FALSE

# setting grid and block dimensions
threads_per_block <- as.integer(1024)
block_dims <- c(threads_per_block, as.integer(1), as.integer(1))
grid_d <- as.integer(ceiling(sqrt((n/n_per_thread)/threads_per_block)))

grid_dims <- c(grid_d, grid_d, as.integer(1))

cat("Grid size:\n")
print(grid_dims)

nthreads <- as.integer(prod(grid_dims)*prod(block_dims))
cat("Total number of threads to launch = ", nthreads, "\n")
if (nthreads*n_per_thread < n){
    stop("Grid is not large enough...!")
}

cat("Running CUDA kernel...\n")

seed = as.integer(0)


tRNGinit <- system.time({
  rng_states <- cudaMalloc(numEls=nthreads, sizeof=as.integer(48), elType="curandState")
  .cuda(setup, rng_states, seed, nthreads, as.integer(verbose), gridDim=grid_dims, blockDim=block_dims)
  cudaDeviceSynchronize()
})

tAlloc <- system.time({
  dX = cudaMalloc(n, sizeof = as.integer(8), elType = "double", strict = TRUE)
  cudaDeviceSynchronize()
})

tCalc <- system.time({
.cuda(rnorm, rng_states, dX, n, mu, sigma, n_per_thread, gridDim=grid_dims, blockDim=block_dims,.numericAsDouble = getOption("CUDA.useDouble", TRUE))
  cudaDeviceSynchronize()
})

tTransferFromGPU <- system.time({
  out = copyFromDevice(obj = dX, nels = dX@nels, type = "double")
  cudaDeviceSynchronize()
})


tCPU <- system.time({
  out2 <- rnorm(n, mu, sigma)
})



cat("RNG initiation time: ", tRNGinit[3], "\n")
cat("GPU memory allocation time: ", tAlloc[3], "\n")
cat("Calculation time (GPU): ", tCalc[3], "\n")
cat("Transfer from GPU time: ", tTransferFromGPU[3], "\n")
cat("Calculation time (CPU): ", tCPU[3], "\n")

