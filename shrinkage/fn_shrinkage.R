##############################################################################################
##############################################################################################
#' @title Posterior Density Function - Example 1
#'
#' @description An example showing how to write a function for use with \code{\link{vegasBayesEvidence}} for
#' Bayesian computation.
#' This example function describes a simple Bayesian hierarchical model comprising of a logistic regression with
#' intercept and single binary covariate for treatment effect each with a hierarchical prior.
#' This has six parameters in total.
#'
#' @details The is an example function written purely in R. It uses a transformation so the density
#' can be integrated across the full domain of each parameter, i.e. the density includes a Jacobian
#'  See \code{vignette("bayes1", package = "vegasr")} for more details.
#'
#' @param theta a numerical matrix of dimension Batch x M, where M is number of parameters, here M=6
#' Batch can be any positive integer
#' @param y a numeric matrix of dimension N x 1, this is the response variable and should be 1.0 or 0.0
#' entries only
#' @param treat a numeric matrix of dimension N x 1, this is the response variable and should be 1.0 or 0.0
#' entries only
#' @param shiftby a numerical scalar used to help avoid underflow. Used in \code{\link{vegasBayesEvidence}}
#' @param uselog a numerical flag value takes either 1.0 or 0.0 and used to return either log or real scale
#' value. Used in \code{\link{vegasBayesEvidence}}
#' @export
## Define log posterior including change of variables
fn_log_post_h1<-function(theta,      # matrix Batch x M
                        y,          # matrix N x 1
                        treat,      # matrix N x 1
                        x,      # matrix N x 1
                        shiftby,    # scalar - no scaling
                        uselog           # scalar - default return exp(log())
){
  
  # this is a trimming function to avoid extreme end of integral limits,
  # same as -inf,+inf
  theta0=pmin(pmax(theta[,1], -0.9999), 0.9999)
  theta1=pmin(pmax(theta[,2], -0.9999), 0.9999)
  theta2=pmin(pmax(theta[,3], -0.9999), 0.9999)
  theta3=pmin(pmax(theta[,4], -0.9999), 0.9999)
  theta4=pmin(pmax(theta[,5], 0.0001), 0.9999)
  theta5=pmin(pmax(theta[,6], 0.0001), 0.9999)
  theta6=pmin(pmax(theta[,7], -0.9999), 0.9999)
  
  jacobianL = (
    log1p(theta0^2) - 2.0*log1p(-(theta0^2))
    + log1p(theta1^2) - 2.0*log1p(-(theta1^2))
    + log1p(theta2^2) - 2.0*log1p(-(theta2^2))
    + log1p(theta3^2) - 2.0*log1p(-(theta3^2))
    + log1p(theta4^2) - 2.0*log1p(-(theta4^2))
    + log1p(theta5^2) - 2.0*log1p(-(theta5^2))
    + log1p(theta6^2) - 2.0*log1p(-(theta6^2))
    
  )
  
  a0=theta0/(1-theta0^2)
  a1=theta1/(1-theta1^2)
  mu0=theta2/(1-theta2^2)
  mu1=theta3/(1-theta3^2)
  sigma0=theta4/(1-theta4^2)
  sigma1=theta5/(1-theta5^2)
  a2=theta6/(1-theta6^2)
  
  # eta=a0+a1*treat + a2*x # (10,3) where a0 and a1 = (3,) and treat is (10,1)
  #                                                broadcasts to (10,3)
  # R doesn't have auto broadcast but this is equivalent
  eta <- sweep(
    treat %*% rbind(a1) + x * a2, # Fixed a2, doesn't need broadcasting
    2,
    a0,
    "+"
  )
  
  logL <- apply(
    sweep(eta, 1, y, "*") - log1p(exp(eta)),
    2,
    sum
  )
  
  # now add the priors
  prior_a0 = stats::dnorm(a0,mean=mu0,sd=sigma0,log=TRUE)
  prior_a1 = stats::dnorm(a1,mean=mu1,sd=sigma1,log=TRUE)
  prior_a2 = stats::dnorm(a2,mean=0,sd=1,log=TRUE)
  prior_mu0 = stats::dnorm(mu0,mean=0.,sd=2.5,log=TRUE)
  prior_mu1 = stats::dnorm(mu1,mean=0.,sd=2.5,log=TRUE)
  prior_sigma0 = extraDistr::dhnorm(sigma0, sigma=2.5,log=TRUE)
  prior_sigma1 = extraDistr::dhnorm(sigma1, sigma=2.5,log=TRUE)
  
  logDens = logL + prior_a0 + prior_a1 + prior_mu0 + prior_mu1 +
    prior_sigma0 + prior_sigma1 + prior_a2
  logPost = logDens + jacobianL
  
  if(uselog==1.){ # search phase for max - keep in log
    return(logPost - shiftby[1])
  } else return(exp(logPost - shiftby[1]) ) # integrand eval - use raw
  
}


if(FALSE){ # to test
  vegasr:::fn_create_data_1(99999)
  # response
  y<-matrix(data=as.numeric(thedata$y),ncol=1)
  # treatment
  treat<-matrix(data=as.numeric(thedata$Treatment),ncol=1)
  # matrix of samole parameter values - nrow is batch, ncol is model dimension
  dummytheta<-matrix(data=c(-0.11,-0.13,0.15,0.11,0.051,0.052,
                            -0.12,-0.1,0.17,0.12,0.052,0.051,
                            -0.13,-0.11,0.11,0.19,0.053,0.054,
                            -0.14, -0.12, 0.12
  ),ncol=7,byrow=TRUE)
  
  ## test if log_posterior works - pass matrix of parameter values
  fn_log_post_h1(theta=dummytheta,y=y,treat=treat,shiftby=0,uselog=1.)
  
}

fn_marg_1_1h<-function(theta,      # matrix Batch x M
                         y,          # matrix N x 1
                         treat,      # matrix N x 1
                         x,      # matrix N x 1
                         shiftby,    # scalar - no scaling
                         uselog           # scalar - default return exp(log())
){
  
  # this is a trimming function to avoid extreme end of integral limits,
  # same as -inf,+inf
  #theta0=pmin(pmax(theta[,1], -0.9999), 0.9999)
  theta1=pmin(pmax(theta[,2-1], -0.9999), 0.9999)
  theta2=pmin(pmax(theta[,3-1], -0.9999), 0.9999)
  theta3=pmin(pmax(theta[,4-1], -0.9999), 0.9999)
  theta4=pmin(pmax(theta[,5-1], 0.0001), 0.9999)
  theta5=pmin(pmax(theta[,6-1], 0.0001), 0.9999)
  theta6=pmin(pmax(theta[,7-1], -0.9999), 0.9999)
  
  jacobianL = (
    #log1p(theta0^2) - 2.0*log1p(-(theta0^2))
    #+ 
    log1p(theta1^2) - 2.0*log1p(-(theta1^2))
    + log1p(theta2^2) - 2.0*log1p(-(theta2^2))
    + log1p(theta3^2) - 2.0*log1p(-(theta3^2))
    + log1p(theta4^2) - 2.0*log1p(-(theta4^2))
    + log1p(theta5^2) - 2.0*log1p(-(theta5^2))
    + log1p(theta6^2) - 2.0*log1p(-(theta6^2))
    
  )
  
  #a0=theta0/(1-theta0^2)
  a0=rep(z,length(theta1)) # z is passed
  a1=theta1/(1-theta1^2)
  mu0=theta2/(1-theta2^2)
  mu1=theta3/(1-theta3^2)
  sigma0=theta4/(1-theta4^2)
  sigma1=theta5/(1-theta5^2)
  a2=theta6/(1-theta6^2)
  
  # eta=a0+a1*treat + a2*x # (10,3) where a0 and a1 = (3,) and treat is (10,1)
  #                                                broadcasts to (10,3)
  # R doesn't have auto broadcast but this is equivalent
  eta <- sweep(
    treat %*% rbind(a1) + x * a2, # Fixed a2, doesn't need broadcasting
    2,
    a0,
    "+"
  )
  
  logL <- apply(
    sweep(eta, 1, y, "*") - log1p(exp(eta)),
    2,
    sum
  )
  
  # now add the priors
  prior_a0 = stats::dnorm(a0,mean=mu0,sd=sigma0,log=TRUE)
  prior_a1 = stats::dnorm(a1,mean=mu1,sd=sigma1,log=TRUE)
  prior_a2 = stats::dnorm(a2,mean=0,sd=1,log=TRUE)
  prior_mu0 = stats::dnorm(mu0,mean=0.,sd=2.5,log=TRUE)
  prior_mu1 = stats::dnorm(mu1,mean=0.,sd=2.5,log=TRUE)
  prior_sigma0 = extraDistr::dhnorm(sigma0, sigma=2.5,log=TRUE)
  prior_sigma1 = extraDistr::dhnorm(sigma1, sigma=2.5,log=TRUE)
  
  logDens = logL + prior_a0 + prior_a1 + prior_mu0 + prior_mu1 +
    prior_sigma0 + prior_sigma1 + prior_a2
  logPost = logDens + jacobianL
  
  if(uselog==1.){ # search phase for max - keep in log
    return(logPost - shiftby[1])
  } else return(exp(logPost - shiftby[1]) ) # integrand eval - use raw
  
}


if(FALSE){ # to test
  vegasr:::fn_create_data_1(99999)
  # response
  y<-matrix(data=as.numeric(thedata$y),ncol=1)
  # treatment
  treat<-matrix(data=as.numeric(thedata$Treatment),ncol=1)
  # matrix of samole parameter values - nrow is batch, ncol is model dimension
  dummytheta<-matrix(data=c(-0.11,-0.13,0.15,0.11,0.051,0.052,
                            -0.12,-0.1,0.17,0.12,0.052,0.051,
                            -0.13,-0.11,0.11,0.19,0.053,0.054
  ),ncol=6,byrow=TRUE)
  
  ## test if log_posterior works - pass matrix of parameter values
  vegasr:::fn_marg_1_1(theta=dummytheta,y=y,treat=treat,shiftby=0,uselog=1.)
  
}