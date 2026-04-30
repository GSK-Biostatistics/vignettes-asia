EXNEX_stan_test2 <- "data {
  int<lower=1> K;                                   // strata
  int<lower=1> N;                                   // subjects
  array[N] int<lower=1, upper=K> k_vec;             // stratum id
  array[N] int<lower=0, upper=1> y;                 // binary outcome
  
  // prior constants (fixed)
  real<lower=0, upper=1> q0;                        // e.g. 0.15
  real<lower=0> nu_mu;                               // SD for mu prior (see note)
  real m;                                            // NEX mean
  real<lower=0> sd;                                  // NEX SD
  
  // optional: Beta prior hyperparameters for pi_k
  real<lower=0> a_pi;                                // e.g. 1
  real<lower=0> b_pi;                                // e.g. 1
}

parameters {
  real mu;
  real<lower=0> sigma;
  
  vector[K] M1;                                     // EX logit per stratum
  vector[K] M2;                                     // NEX logit per stratum
  
  vector<lower=0, upper=1>[K] pi;                   // stratum-specific mixing weight
}

model {
  // Priors
  mu    ~ normal(logit(q0), nu_mu);                  // nu_mu is SD
  sigma ~ normal(0, 1);                              // example g(.); adjust as needed
  
  M1 ~ normal(mu, sigma);                            // EX
  M2 ~ normal(m, sd);                                // NEX
  
  pi ~ beta(a_pi, b_pi);                             // mixing weights
  
  // Subject-level mixture likelihood
  for (i in 1:N) {
    int k = k_vec[i];
    target += log_mix(
      pi[k],
      bernoulli_logit_lpmf(y[i] | M1[k]),
      bernoulli_logit_lpmf(y[i] | M2[k])
    );
  }
}

generated quantities {
  vector[N] log_lik;
  
  matrix[K, 2] post_prob;                            // Pr(EX|data_k), Pr(NEX|data_k)
  array[K] int<lower=1, upper=2> class_k;
  
  vector[K] p_ex;
  vector[K] p_nex;
  vector[K] p_mix;
  
  // accumulate stratum-level log-likelihood under each component
  vector[K] ll_ex  = rep_vector(0.0, K);
  vector[K] ll_nex = rep_vector(0.0, K);
  
  // pointwise log-lik + accumulate per stratum
  for (i in 1:N) {
    int k = k_vec[i];
    real lp_ex_i  = bernoulli_logit_lpmf(y[i] | M1[k]);
    real lp_nex_i = bernoulli_logit_lpmf(y[i] | M2[k]);
    
    log_lik[i] = log_mix(pi[k], lp_ex_i, lp_nex_i);
    
    ll_ex[k]  += lp_ex_i;
    ll_nex[k] += lp_nex_i;
  }
  
  // posterior responsibilities per stratum
  for (k in 1:K) {
    vector[2] lp;
    vector[2] pr;
    
    lp[1] = log(pi[k])   + ll_ex[k];
    lp[2] = log1m(pi[k]) + ll_nex[k];
    pr = softmax(lp);
    
    post_prob[k,1] = pr[1];
    post_prob[k,2] = pr[2];
    class_k[k] = categorical_rng(pr);
    
    p_ex[k]  = inv_logit(M1[k]);
    p_nex[k] = inv_logit(M2[k]);
    p_mix[k] = pr[1] * p_ex[k] + pr[2] * p_nex[k];
  }
}"



### Run the model

# Create list with input values for Stan model
data_input_1 <- list(
  K = K,                # number of subgroups
  N = N,                # total sample size
  N_k = N_k,            # sample size per basket
  k_vec = k_vec,        # N x 1 vector of subgroup indicators
  z_vec = z_vec,        # N x 1 vector of active treatment arm indicators
  y = y,
  q0 = 0.15,
  nu_mu = 10,
  m = -0.62,
  sd = 4.4,
  a_pi= 2,
  b_pi = 12
)



### Compile and fit model
options(mc.cores = parallel::detectCores())
# Compile model (only run the line below once for a simulation study - compilation not dependent on any
# simulation inputs as defined by scenarios or on simulated data)
stan_mod_1 <- stan_model(model_code = EXNEX_stan_test2)
# Fit model
nsamps <- 30000       # number of posterior samples (after removing burn-in) per chain
nburnin <- 1000       # number of burn-in samples to remove at beginning of each chain
nchains <- 4          # number of chains
#BHM_pars_1 <- c("beta","mu0", "sigma0","mu1", "sigma1", "beta","beta_tr")     # parameters to sample
start_time <- Sys.time()
stan_fit_2 <- sampling(stan_mod_1, data = data_input_1, 
                       #pars = BHM_pars_1,
                       iter = nsamps + nburnin, warmup = nburnin, chains = nchains)

end_time <- Sys.time()
end_time - start_time
post_draws <- as.matrix(stan_fit_2)           # posterior samples of each parameter


with(thedata, tapply(y, list(basketID, Treatment), mean))
###     0    1
# 1 0.20 0.32
# 2 0.14 0.28
# 3 0.10 0.44
# 4 0.22 0.40