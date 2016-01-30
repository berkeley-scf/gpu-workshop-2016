extern "C"
__global__ void compute_probs(double* alphas, double* rands, double* probs, int n, int K, int M) {
    // assign overall id/index of the thread = id of row
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int threads_per_block = blockDim.x; 

    // set up shared memory: half for probs and half for w
    extern __shared__ double shared[];
    double* probs_shared = shared;
    double* w = &shared[K*threads_per_block];    // shared mem is one big block, so need to index into latter portion of it to use for w


    if(i < n) {
      double maxval;    
      int m, k;
      int maxind;
      double M_d = (double) M; 
      
      // initialize shared memory probs
      for(k = 0; k < K; ++k) {
        probs_shared[k*threads_per_block + threadIdx.x] = 0.0;
      }

      for(m = 0; m < M; ++m){     // loop over Monte Carlo iterations 
        for(k = 0; k < K; ++k){   // generate W ~ N(alpha, 1)
          w[k*threads_per_block + threadIdx.x] = alphas[k*n + i] + rands[k*M + m];
        }
        maxind = K-1;
        maxval = w[(K-1)*threads_per_block + threadIdx.x];
        for(k = 0; k < (K-1); ++k){
          if(w[k*threads_per_block + threadIdx.x] > maxval){
            maxind = k;
            maxval = w[k*threads_per_block + threadIdx.x];
          } 
        }
        probs_shared[maxind*threads_per_block + threadIdx.x] += 1.0;
      }
      for(k = 0; k < K; ++k) {
        probs_shared[k*threads_per_block + threadIdx.x] /= M_d;
      }
      
      // copy to device memory so can be returned to CPU
      for(k = 0; k < K; ++k) {
          probs[k*n + i] = probs_shared[k*threads_per_block + threadIdx.x];
      }
    }

}
