stan_test2 <- "data {
  int<lower=1> K;
  int<lower=1> N;
  array[N] int<lower=1, upper=K> k_vec;
  array[N] int<lower=0, upper=1> z_vec;
  array[N] int<lower=0, upper=1> y;

  vector<lower=0>[2] aMix;

  // NEX scales (fixed)
  real<lower=0> tau_alpha_nex;
  real<lower=0> tau_delta_nex;
}

parameters {
  simplex[2] pMix;

  real mu_alpha;
  real<lower=0> tau_alpha_ex;

  real mu_delta;
  real<lower=0> tau_delta_ex;

  real<lower=0> kappa_alpha;   // inflation factor
  real<lower=0> kappa_delta;

  vector[K] alpha;
  vector[K] delta;
}

transformed parameters {
  real<lower=0> tau_alpha_nex = tau_alpha_ex * (1 + kappa_alpha);
  real<lower=0> tau_delta_nex = tau_delta_ex * (1 + kappa_delta);
}

model {
// mixture weight prior
pMix ~ dirichlet(aMix);  // keep, but consider aMix changes below

// location priors (logit scale)
mu_alpha ~ normal(0, 1.5);   // intercept: typical weakly-informative
mu_delta ~ normal(0, 1.0);   // treatment log-OR often tighter than intercept

// scale priors: half-t is usually more robust than half-normal
tau_alpha_ex ~ student_t(3, 0, 1.0);   // half-t due to <lower=0>
tau_delta_ex ~ student_t(3, 0, 0.5);

// enforce NEX wider than EX via kappa priors
kappa_alpha ~ lognormal(log(2), 0.5);  // median ~2, allows bigger
kappa_delta ~ lognormal(log(2), 0.5);

  // EXNEX mixture priors (same pMix for alpha and delta)
  for (k in 1:K) {
    target += log_mix(pMix[1],
                      normal_lpdf(alpha[k] | mu_alpha, tau_alpha_ex),
                      normal_lpdf(alpha[k] | mu_alpha, tau_alpha_nex));

    target += log_mix(pMix[1],
                      normal_lpdf(delta[k] | mu_delta, tau_delta_ex),
                      normal_lpdf(delta[k] | mu_delta, tau_delta_nex));
  }

  // likelihood
  for (i in 1:N) {
    y[i] ~ bernoulli_logit(alpha[k_vec[i]] + delta[k_vec[i]] * z_vec[i]);
  }
}

generated quantities {
  vector[K] p_control = inv_logit(alpha);
  vector[K] p_active  = inv_logit(alpha + delta);

  vector[K] pr_ex_alpha;
  vector[K] pr_ex_delta;

  for (k in 1:K) {
    // posterior Pr(EX | alpha[k], hyperparams) for this draw
    real lp_ex_a  = log(pMix[1]) + normal_lpdf(alpha[k] | mu_alpha, tau_alpha_ex);
    real lp_nex_a = log(pMix[2]) + normal_lpdf(alpha[k] | mu_alpha, tau_alpha_nex);
    pr_ex_alpha[k] = inv_logit(lp_ex_a - lp_nex_a);

    // posterior Pr(EX | delta[k], hyperparams) for this draw
    real lp_ex_d  = log(pMix[1]) + normal_lpdf(delta[k] | mu_delta, tau_delta_ex);
    real lp_nex_d = log(pMix[2]) + normal_lpdf(delta[k] | mu_delta, tau_delta_nex);
    pr_ex_delta[k] = inv_logit(lp_ex_d - lp_nex_d);
  }
}"

#####

# Create list with input values for Stan model
data_input_1 <- list(
  K = K,                # number of subgroups
  N = N,                # total sample size
  N_k = N_k,            # sample size per basket
  k_vec = k_vec,        # N x 1 vector of subgroup indicators
  z_vec = z_vec,        # N x 1 vector of active treatment arm indicators
  y = y,                 # N x 1 vector of binary responses
  aMix = c(2, 2),
  tau_alpha_nex = 1.5,
  tau_delta_nex = 1.5
)



### Compile and fit model
options(mc.cores = parallel::detectCores())
# Compile model (only run the line below once for a simulation study - compilation not dependent on any
# simulation inputs as defined by scenarios or on simulated data)
stan_mod_1 <- stan_model(model_code = stan_test2)
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
post_draws <- as.matrix(stan_fit_2)   