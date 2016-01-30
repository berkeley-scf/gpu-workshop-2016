M <- 1e3

source('setup_calc.R')

props2 <- props3 <- props

# 2-3 sec per iteration for M = 1e6
# 72 sec for M=1000
system.time({
for(i in 1:n) {
    tmp <- alphas[ , i] + rands
    id <- apply(tmp, 2, which.max)
    tbl <- table(id)
    props[as.integer(names(tbl)) , i] <- tbl / n
    if(i %% 1000 == 0) print(c(i, date()))
}
})

# 57 sec for M=1000
system.time({
for(i in 1:n) {
    tmp <- t(alphas[ , i] + rands)
    id <- rep(1, M)
    for(k in 2:K) {
        wh <- tmp[, k ] > tmp[ , 1 ]
        id[wh] <- k
        tmp[wh, 1] <- tmp[wh, k ]
    }
    tbl <- table(id)
    props2[as.integer(names(tbl)) , i] <- tbl / n
    if(i %% 1000 == 0) print(c(i, date()))
}
})

nProc <- 4 

library(doParallel)
registerDoParallel(nProc)

# 29 sec for M=1000, 4 cores
# 14 sec for M=1000, 8 cores
system.time({
props3 <- foreach(i = 1:n, .combine = cbind) %dopar% {
    tmp <- t(alphas[ , i] + rands)
    id <- rep(1, M)
    for(k in 2:K) {
        wh <- tmp[, k ] > tmp[ , 1 ]
        id[wh] <- k
        tmp[wh, 1] <- tmp[wh, k ]
    }
    tbl <- table(id)
    out <- rep(0, K)
    out[as.integer(names(tbl))] <- tbl / M
        if(i %% 1000 == 0) print(c(i, date()))
    out
}
})
