import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz
from scipy.stats import binom


np.random.seed(123)  # Set the random seed for reproducibility
trial = 4
theta_real = 0.35  # Unknown value in a real experiment

# Generate random variates from a Binomial distribution
data = binom(n=1, p=theta_real).rvs(trial)
# print(data)



with pm.Model() as our_first_model:
    θ = pm.Beta('θ', alpha=1., beta=1.)
    y = pm.Bernoulli('y', p=θ, observed=data)
    idata = pm.sample(1000, random_seed=4591)


