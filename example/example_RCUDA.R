# modification of one of the RCUDA examples to use use double precision

library(RCUDA)

if(!exists('unitStrides') || is.null(unitStrides)) unitStrides <- FALSE
if(!exists('sharedMem') || is.null(sharedMem)) sharedMem <- FALSE
if(!exists('float') || is.null(float)) float <- FALSE

M <- as.integer(1e4) # important to have as integer!

# get the alphas and generate the random numbers
source('setup_calc.R')

cat("Setting cuGetContext(TRUE)...\n")
cuGetContext(TRUE)

# compile the kernel into a form that RCUDA can load; equivalent to this nvcc call:
# system("nvcc --ptx  -arch=compute_20 -code=sm_20,compute_20 -o compute_probs.ptx compute_probs.cu")

fn <- "compute_probs"
if(unitStrides) fn <- paste0(fn, "_unitStrides")
if(sharedMem) fn <- paste0(fn, "_sharedMem")
if(float) fn <- paste0(fn, "_float")
ptx = nvcc(file = paste0(fn, ".cu"), out = 'compute_probs.ptx',
  target = "ptx", "-arch=compute_20", "-code=sm_20,compute_20")

mod = loadModule(ptx)
compute_probs = mod$compute_probs

# setting grid and block dimensions
threads_per_block <- as.integer(192)
if(sharedMem) threads_per_block <- as.integer(96) # need fewer threads so that have enough room in 48Kb of shared memory 
block_dims <- c(threads_per_block, as.integer(1), as.integer(1))
grid_d <- as.integer(ceiling(n/threads_per_block))

grid_dims <- c(grid_d, as.integer(1), as.integer(1))

cat("Grid size:\n")
print(grid_dims)

nthreads <- prod(grid_dims)*prod(block_dims)
cat("Total number of threads to launch = ", nthreads, "\n")
if (nthreads < n){
    stop("Grid is not large enough...!")
}

cat("Running CUDA kernel...\n")

if(unitStrides) {
    probs <- matrix(0, nrow = n, ncol = K)
    tmp <- matrix(0, nrow = n, ncol = K)
    rands <- t(rands)
    alphas <- t(alphas)
} else {
    probs <- matrix(0, nrow = K, ncol = n)
    tmp <- matrix(0, nrow = K, ncol = n)
}

if(!float) {
    strict = TRUE # for double
    cuCtxSetSharedMemConfig("CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE")
} else strict = FALSE

sharedMemSize <- as.integer(
    ifelse(float, 4, 8)*K*threads_per_block*2
    )

if(sharedMem && sharedMemSize > 48000) stop("trying to use too much shared memory")

# basic usage with manual transfer
tTransferToGPU <- system.time({
  devAlphas = copyToDevice(alphas, strict = strict)
  devRands = copyToDevice(rands, strict = strict)
  devProbs = copyToDevice(probs, strict = strict)
  cudaDeviceSynchronize()
})
tCalc <- system.time({
    if(float) {
        .cuda(compute_probs, devAlphas, devRands, devProbs,
              n, K, M, gridDim = grid_dims, blockDim = block_dims, sharedMemBytes = ifelse(sharedMem, sharedMemSize, as.integer(0)))
    } else
        .cuda(compute_probs, devAlphas, devRands, devProbs,
      n, K, M, gridDim = grid_dims, blockDim = block_dims, sharedMemBytes = ifelse(sharedMem, sharedMemSize, as.integer(0)), .numericAsDouble = getOption("CUDA.useDouble", TRUE))
  cudaDeviceSynchronize()
})
tTransferFromGPU <- system.time({
  out = copyFromDevice(obj = devProbs, nels = devProbs@nels, type = "double")
  cudaDeviceSynchronize()
})


cat("Input values: ", alphas[1:3], "\n")
cat("Output values: ", out[1:3], "\n")
                     
cat("Transfer to GPU time: ", tTransferToGPU[3], "\n")
cat("Calculation time (GPU): ", tCalc[3], "\n")
cat("Transfer from GPU time: ", tTransferFromGPU[3], "\n")




