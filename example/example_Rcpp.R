M <- 1e4

source('setup_calc.R')

require(Rcpp)
require(inline)

cppFunction('
  NumericMatrix compute_probs_mp(NumericMatrix alpha, NumericMatrix rands, int M, int n, int K, int nProc){

    NumericMatrix probs(n, K);
    int i;

    omp_set_num_threads(nProc);

    #pragma omp parallel for  
    for(i = 0; i < n; ++i){
      double max;    
      int m, k;
      int maxind;
      NumericVector rvVals(K);

      for(k = 0; k < K; ++k){
         probs(i,k) = 0.0;
      }
      for(m = 0; m < M; ++m){
        for(k = 0; k < K; ++k){
          rvVals(k) = alpha(i, k) + rands(m, k);
        }
        maxind = K-1;
        max = rvVals(K-1);
        for(k = 0; k < (K-1); ++k){
          if(rvVals(k) > max){
            maxind = k;
            max = rvVals(k);
          } 
        }
        probs(i,maxind) += 1.0;
      }
      for(k = 0; k < K; ++k) {
        probs(i,k) /= M;
      }
    }
    return probs;
  }
', plugins = c("openmp"), includes = c('#include <omp.h>'))

alphas <- t(alphas)
rands <- t(rands)

# 47 sec for 10000
system.time({          
    props1 <- compute_probs_mp(alphas, rands, M, n, K, nProc = 1)
})
# 11.9 sec for 10000
system.time({
    props2 <- compute_probs_mp(alphas, rands, M, n, K, nProc = 4)  
})
# 6.0 sec for 10000
system.time({
    props3 <- compute_probs_mp(alphas, rands, M, n, K, nProc = 8)  
})

# using transposed alpha,rands doesn't change things much: 

cppFunction('
  NumericMatrix compute_probs_mp2(NumericMatrix alpha, NumericMatrix rands, int M, int n, int K, int nProc){

    NumericMatrix probs(K, n);
    int i;

    omp_set_num_threads(nProc);

    #pragma omp parallel for  
    for(i = 0; i < n; ++i){
      double max;    
      int k, m;
      int maxind;
      NumericVector rvVals(K);

      for(k = 0; k < K; ++k){
         probs(k,i) = 0.0;
      }
      for(m = 0; m < M; ++m){
        for(k = 0; k < K; ++k){
          rvVals(k) = alpha(k,i) + rands(k, m);
        }
        maxind = K-1;
        max = rvVals(K-1);
        for(k = 0; k < (K-1); ++k){
          if(rvVals(k) > max){
            maxind = k;
            max = rvVals(k);
          } 
        }
        probs(maxind, i) += 1.0;
      }
      for(k = 0; k < K; ++k) {
        probs(k,i) /= M;
      }
    }
    return probs;
  }
', plugins = c("openmp"), includes = c('#include <omp.h>'))

alphas <- t(alphas)
rands <- t(rands)

# 50 sec. for 10000
system.time({          
    props1 <- compute_probs_mp2(alphas, rands, M, n, K, nProc = 1)
})
# 12.3 sec  for 10000
system.time({
    props2 <- compute_probs_mp(alphas, rands, M, n, K, nProc = 4)  
})
# 6.2 sec  for 10000
system.time({
    props3 <- compute_probs_mp(alphas, rands, M, n, K, nProc = 8)  
})


