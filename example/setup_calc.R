alphas <- t(as.matrix(read.csv('alphas.csv', header = FALSE)))

n <- ncol(alphas)
K <- nrow(alphas)

props  <- matrix(0, K, n)

set.seed(0)

system.time({
rands <- matrix(rnorm(M*K), nrow = K, ncol = M)
})

