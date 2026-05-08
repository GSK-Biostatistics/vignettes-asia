### How outrageous are yoiur priors

### Data simulation

### Prepare model inputs
set.seed(99999)
# Set up data
rr_k_ctrl <- c(0.20)        # control response rate for each basket
rr_k_trt <- c(0.40)         # treatment response rate for each basket

K<-length(rr_k_ctrl)        # number of baskets

N_k_ctrl <- rep(100, K)     # number of control participants per basket
N_k_trt <- rep(100, K)      # number of treatment participants per basket
N_k <- N_k_ctrl + N_k_trt   # number of participants per basket (both arms combined)
N <- sum(N_k)               # total sample size
k_vec <- rep(1:K, N_k)      # N x 1 vector of basket indicators (1 to K)

z_vec<-NULL;
y<-NULL;
for(i in 1:K){ # for each basket repeat 0-control 1-trt according to the specifc Ns
  z_vec<-c(z_vec,rep(0:1,c(N_k_ctrl[i],N_k_trt[i]))) # treatment/control indicator
  y<-c(y,
       c(rbinom(N_k_ctrl[i],1,rr_k_ctrl[i]), # bernoulli for control
         rbinom(N_k_trt[i],1,rr_k_trt[i]))) #           for trt
}

thedata<-data.frame(y,basketID=k_vec,Treatment=z_vec)

### Dummy model
N <- 1000
sd <- 10

mu0 <- rnorm(N, 0, sd)
sigma0 <- abs(rnorm(N, 0, sd))
mu1 <- rnorm(N, 0, sd)
sigma1 <- abs(rnorm(N, 0, sd))
theta <- matrix(rnorm(N *2), nrow = N, ncol = 2) # For non-centerd

##Vectorized calculation

beta0 <- mu0 + sigma0*theta[,1]
beta1 <- mu1 + sigma1*theta[,2]
#eta <- beta0 + beta1*z_vec #Linear predictor
# p <- plogis(eta)
# y <- rbinom(N, 1, p)

### From McElreath

p <- sapply(z_vec, function(z) 
  plogis(beta0 + beta1*z))

y <- matrix(rbinom(length(p), 1, p), nrow = nrow(p), ncol = ncol(p))

plot(z_vec, p[1,], ylim = c(0, 1))


### Gaussian data

### Prepare model inputs
set.seed(99999)
# Set up data
rr_k_ctrl <- c(0.20)        # control response rate for each basket
rr_k_trt <- c(0.40)         # treatment response rate for each basket

K<-length(rr_k_ctrl)        # number of baskets

N_k_ctrl <- rep(100, K)     # number of control participants per basket
N_k_trt <- rep(100, K)      # number of treatment participants per basket
N_k <- N_k_ctrl + N_k_trt   # number of participants per basket (both arms combined)
N <- sum(N_k)               # total sample size
k_vec <- rep(1:K, N_k)      # N x 1 vector of basket indicators (1 to K)

z_vec<-NULL;
y<-NULL;
for(i in 1:K){ # for each basket repeat 0-control 1-trt according to the specifc Ns
  z_vec<-c(z_vec,rep(0:1,c(N_k_ctrl[i],N_k_trt[i]))) # treatment/control indicator
  y<-c(y,
       c(rnorm(N_k_ctrl[i],1,rr_k_ctrl[i]), # bernoulli for control
         rnorm(N_k_trt[i],1,rr_k_trt[i]))) #           for trt
}

thedata<-data.frame(y,basketID=k_vec,Treatment=z_vec)
