extern "C"
__global__ void compute_probs(double* alphas, double* rands, double* probs, int n, int K, int M) {
    // assign overall id/index of the thread = id of row
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if(i < n) {
      double maxval;    
      int m, k;
      int maxind;
      double M_d = (double) M; 
      double* w = new double[K];

      for(k = 0; k < K; ++k){   // initialize probs (though already done on CPU)
         probs[i*K + k] = 0.0;
      }

      // core computations
      for(m = 0; m < M; ++m){   // loop over Monte Carlo iterations
        for(k = 0; k < K; ++k){  // generate W ~ N(alpha, 1)
          w[k] = alphas[i*K + k] + rands[m*K + k];
        }

        // determine which category has max W
        maxind = K-1;
        maxval = w[K-1];
        for(k = 0; k < (K-1); ++k){
          if(w[k] > maxval){
            maxind = k;
            maxval = w[k];
          } 
        }
        probs[i*K + maxind] += 1.0;
      }

      // compute final proportions
      for(k = 0; k < K; ++k) {
        probs[i*K + k] /= M_d;
      }
      free(w);
    }

}
