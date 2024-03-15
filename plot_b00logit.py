"""

Base model
==========

Logit model.

:author: Michel Bierlaire, EPFL
:date: Thu Jul 13 16:18:10 2023

"""
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, log, exp
import numpy as np
import scipy.optimize as so
from scipy.special import logsumexp


def LL_MNL(params, Att_list, ch, av): # Log MNL
    V_list = []
    j = 0
    for i in range(len(Att_list)):
        att = Att_list[i]
        V_list += [np.matmul(att, params[j:j+att.shape[1]])]
        j += att.shape[1]
    V_list = np.array(V_list).T
    ls = logsumexp(V_list+av, axis = 1)
    P_ch = np.array([eV[i][ch[i]] for i in range(len(ch))])
    
    LL_i = np.array([V_list[i, ch[i]] - ls[i] for i in range(len(ch))])
    LL = np.sum(LL_i)    
    return (-1)*LL

def LL_RDM(params, RL_dict, ch_dict): 
    V_list = []
    j = 0
    for i in range(len(Att_list)):
        att = Att_list[i]
        V_list += [np.matmul(att, params[j:j+att.shape[1]])]
        j += att.shape[1]
    V_list = np.array(V_list).T
    V_list = V_list-np.min(V_list, axis = 1)[:, None]
    eV = np.exp(V_list)*av
    eV /= np.sum(eV, axis = 1)[:, None]
    P_ch = np.array([eV[i][ch[i]] for i in range(len(ch))])
    print(len(P_ch > 0))
    LL = np.sum(np.log(P_ch[P_ch > 0]))
    return (-1)*LL

    P = np.zeros(len(RL_dict.keys()))
    i = 0
    for OD in RL_dict.keys():
        ch = ch_dict[OD]
        att = RL_dict[OD]
        av = att[:, 0]
        att = att[:, 1:] - np.min(att[:, 1:], axis = 0)
        logit = np.exp(params*att)
        logit /= np.sum(logit, axis = 0)
        logit_product = av*(1-np.prod(1-logit, axis = 1))
        P[i] = logit_product[ch]/sum(logit_product)
        i += 1
    return P

def LL_GRDM(params, RL_dict, ch_dict): #Possibility of returning the gradient of the Logarithm
    P = np.zeros(len(RL_dict.keys()))
    n = len(params)
    scales = params[n//2:]
    params = params[:n//2]
    i = 0
    for OD in RL_dict.keys():
        ch = ch_dict[OD]
        att = RL_dict[OD]
        av = att[:, 0]
        att = att[:, 1:] - np.min(att[:, 1:], axis = 0)
        logit = np.exp(params*att)
        logit /= np.sum(logit, axis = 0)
        logit_product = av*(1-np.prod((1-logit)**scales, axis = 1))
        P[i] = logit_product[ch]/sum(logit_product)
        i += 1
    return P

def callbackF(x):
    print('Current parameter estimates:', [round(elem, 3) for elem in x])
    print('LL=', fun(x, Att_list, ch, av))

def estim(fun, params_0, bnds, *args): # Fun is the likelihood function to minimize
    par_estimators = so.minimize(fun, params_0, args=(args), method=None,
                                 jac=None, hess=None, hessp=None, bounds=bnds, constraints=(), tol=1e-12, callback=callbackF, options=None)
    estimates = np.append(par_estimators.x, par_estimators.fun)
    return estimates

#%%

x = pd.read_csv('swissmetro.dat', sep = '\t')
include = ((x['PURPOSE'] != 1) * (x['PURPOSE'] != 3) + (x['CHOICE'] == 0)) <= 0
x = x[include]

Att_train = x[['TRAIN_AV', 'TRAIN_TT', 'TRAIN_CO']]
Att_sm = x[['SM_TT', 'SM_CO']]
Att_car = x[['CAR_AV', 'CAR_TT', 'CAR_CO']]

Att_list = [Att_train, Att_sm, Att_car]
av = np.array(x[['TRAIN_AV', 'SM_AV', 'CAR_AV']])
ch = np.array(x['CHOICE']-1, dtype=int)
av2 = np.nan_to_num((-np.inf)*(1-av))

#%% MNL
fun = LL_MNL
params_0 = [0]*8

bnds = [[None, None]]*8

est_mnl = estim(fun, params_0, bnds, Att_list, ch, av2)

# %%
# See :ref:`swissmetro_data`
from swissmetro_data import (
    database,
    CHOICE,
    SM_AV,
    CAR_AV_SP,
    TRAIN_AV_SP,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
)

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b00logit'
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False

# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood(av)

# Estimate the parameters
results = the_biogeme.estimate()

print(results.short_summary())

# Get the results in a pandas table
pandas_results = results.getEstimatedParameters()
print(pandas_results)

# %% RDM

V1 = log( 1 - (1 - exp(B_TIME  * TRAIN_TT_SCALED)/(exp(B_TIME  * TRAIN_TT_SCALED) +
                                                   exp(B_TIME  * SM_TT_SCALED) + exp(B_TIME  * CAR_TT_SCALED))) *
         (1 - exp(B_COST  * TRAIN_COST_SCALED)/(exp(B_COST  * TRAIN_COST_SCALED) +
                                                            exp(B_COST  * SM_COST_SCALED) + exp(B_COST  * CAR_CO_SCALED))))


V2 = log( 1 - (1 - exp(B_TIME  * SM_TT_SCALED)/(exp(B_TIME  * TRAIN_TT_SCALED) +
                                                   exp(B_TIME  * SM_TT_SCALED) + exp(B_TIME  * CAR_TT_SCALED))) *
         (1 - exp(B_COST  * SM_COST_SCALED)/(exp(B_COST  * TRAIN_COST_SCALED) +
                                                            exp(B_COST  * SM_COST_SCALED) + exp(B_COST  * CAR_CO_SCALED))))

V3 = log( 1 - (1 - exp(B_TIME  * CAR_TT_SCALED)/(exp(B_TIME  * TRAIN_TT_SCALED) +
                                                   exp(B_TIME  * SM_TT_SCALED) + exp(B_TIME  * CAR_TT_SCALED))) *
         (1 - exp(B_COST  * CAR_CO_SCALED)/(exp(B_COST  * TRAIN_COST_SCALED) +
                                                            exp(B_COST  * SM_COST_SCALED) + exp(B_COST  * CAR_CO_SCALED))))


V = {1: V1, 2: V2, 3: V3}


# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}


# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)


# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b00rdm'
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False


# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood(av)


# Estimate the parameters
results = the_biogeme.estimate()


print(results.short_summary())


# Get the results in a pandas table
pandas_results = results.getEstimatedParameters()
pandas_results

# %% GRDM

LAMBDA_TIME = Beta('LAMBDA_TIME', 1, 0.01, 10, 0)
LAMBDA_COST = Beta('LAMBDA_COST', 1, 0.01, 10, 0)


V1 = log( 1 - (1 - exp(B_TIME  * TRAIN_TT_SCALED)/(exp(B_TIME  * TRAIN_TT_SCALED) +
                                                   exp(B_TIME  * SM_TT_SCALED) + exp(B_TIME  * CAR_TT_SCALED)))**LAMBDA_TIME *
         (1 - exp(B_COST  * TRAIN_COST_SCALED)/(exp(B_COST  * TRAIN_COST_SCALED) +
                                                            exp(B_COST  * SM_COST_SCALED) + exp(B_COST  * CAR_CO_SCALED)))**LAMBDA_COST)


V2 = log( 1 - (1 - exp(B_TIME  * SM_TT_SCALED)/(exp(B_TIME  * TRAIN_TT_SCALED) +
                                                   exp(B_TIME  * SM_TT_SCALED) + exp(B_TIME  * CAR_TT_SCALED)))**LAMBDA_TIME *
         (1 - exp(B_COST  * SM_COST_SCALED)/(exp(B_COST  * TRAIN_COST_SCALED) +
                                                            exp(B_COST  * SM_COST_SCALED) + exp(B_COST  * CAR_CO_SCALED)))**LAMBDA_COST)

V3 = log( 1 - (1 - exp(B_TIME  * CAR_TT_SCALED)/(exp(B_TIME  * TRAIN_TT_SCALED) +
                                                   exp(B_TIME  * SM_TT_SCALED) + exp(B_TIME  * CAR_TT_SCALED)))**LAMBDA_TIME *
         (1 - exp(B_COST  * CAR_CO_SCALED)/(exp(B_COST  * TRAIN_COST_SCALED) +
                                                            exp(B_COST  * SM_COST_SCALED) + exp(B_COST  * CAR_CO_SCALED)))**LAMBDA_COST)

#%%

V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b00grdm'
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False

# %%
# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood(av)

# %%
# Estimate the parameters
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = results.getEstimatedParameters()
pandas_results

#%% MNL + GRDM

#MNL parameters
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME_MNL = Beta('B_TIME_MNL', 0, None, None, 0)
B_COST_MNL = Beta('B_COST_MNL', 0, None, None, 0)

#GRDM parameters
LAMBDA_TIME = Beta('LAMBDA_TIME', 1, 0, 10, 0)
LAMBDA_COST = Beta('LAMBDA_COST', 1, 0, 10, 0)

B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# MU
mu = Beta('mu', 0, None, None, 0)

# Definition of the utility functions.
V1_MNL = ASC_TRAIN + B_TIME_MNL * TRAIN_TT_SCALED + B_COST_MNL * TRAIN_COST_SCALED
V2_MNL = B_TIME_MNL * SM_TT_SCALED + B_COST_MNL * SM_COST_SCALED
V3_MNL = ASC_CAR + B_TIME_MNL * CAR_TT_SCALED + B_COST_MNL * CAR_CO_SCALED

V1_GRDM = log( 1 - (1 - exp(B_TIME  * TRAIN_TT_SCALED)/(exp(B_TIME  * TRAIN_TT_SCALED) +
                                                   exp(B_TIME  * SM_TT_SCALED) + exp(B_TIME  * CAR_TT_SCALED)))**LAMBDA_TIME *
         (1 - exp(B_COST  * TRAIN_COST_SCALED)/(exp(B_COST  * TRAIN_COST_SCALED) +
                                                            exp(B_COST  * SM_COST_SCALED) + exp(B_COST  * CAR_CO_SCALED)))**LAMBDA_COST)


V2_GRDM = log( 1 - (1 - exp(B_TIME  * SM_TT_SCALED)/(exp(B_TIME  * TRAIN_TT_SCALED) +
                                                   exp(B_TIME  * SM_TT_SCALED) + exp(B_TIME  * CAR_TT_SCALED)))**LAMBDA_TIME *
         (1 - exp(B_COST  * SM_COST_SCALED)/(exp(B_COST  * TRAIN_COST_SCALED) +
                                                            exp(B_COST  * SM_COST_SCALED) + exp(B_COST  * CAR_CO_SCALED)))**LAMBDA_COST)

V3_GRDM = log( 1 - (1 - exp(B_TIME  * CAR_TT_SCALED)/(exp(B_TIME  * TRAIN_TT_SCALED) +
                                                   exp(B_TIME  * SM_TT_SCALED) + exp(B_TIME  * CAR_TT_SCALED)))**LAMBDA_TIME *
         (1 - exp(B_COST  * CAR_CO_SCALED)/(exp(B_COST  * TRAIN_COST_SCALED) +
                                                            exp(B_COST  * SM_COST_SCALED) + exp(B_COST  * CAR_CO_SCALED)))**LAMBDA_COST)

ProbMNL = exp(mu)/(1+exp(mu))
ProbGRDM = 1 - ProbMNL

V_MNL = {1: V1_MNL, 2: V2_MNL, 3: V3_MNL}

V_GRDM = {1: V1_GRDM, 2: V2_GRDM, 3: V3_GRDM}

prob_MNL = models.logit(V_MNL, av, CHOICE)
prob_GRDM = models.logit(V_GRDM, av, CHOICE)
prob = ProbMNL*prob_MNL + ProbGRDM*prob_GRDM
logprob = log(prob)

# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'MNL+GRDM'
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False

# %%
# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood(av)

# %%
# Estimate the parameters
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = results.getEstimatedParameters()
pandas_results

#%% MNL + RDM

#MNL parameters
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME_MNL = Beta('B_TIME_MNL', 0, None, None, 0)
B_COST_MNL = Beta('B_COST_MNL', 0, None, None, 0)

B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# MU
mu = Beta('mu', 0, None, None, 0)

# Definition of the utility functions.
V1_MNL = ASC_TRAIN + B_TIME_MNL * TRAIN_TT_SCALED + B_COST_MNL * TRAIN_COST_SCALED
V2_MNL = B_TIME_MNL * SM_TT_SCALED + B_COST_MNL * SM_COST_SCALED
V3_MNL = ASC_CAR + B_TIME_MNL * CAR_TT_SCALED + B_COST_MNL * CAR_CO_SCALED

V1_GRDM = log( 1 - (1 - exp(B_TIME  * TRAIN_TT_SCALED)/(exp(B_TIME  * TRAIN_TT_SCALED) +
                                                   exp(B_TIME  * SM_TT_SCALED) + exp(B_TIME  * CAR_TT_SCALED))) *
         (1 - exp(B_COST  * TRAIN_COST_SCALED)/(exp(B_COST  * TRAIN_COST_SCALED) +
                                                            exp(B_COST  * SM_COST_SCALED) + exp(B_COST  * CAR_CO_SCALED))))


V2_GRDM = log( 1 - (1 - exp(B_TIME  * SM_TT_SCALED)/(exp(B_TIME  * TRAIN_TT_SCALED) +
                                                   exp(B_TIME  * SM_TT_SCALED) + exp(B_TIME  * CAR_TT_SCALED))) *
         (1 - exp(B_COST  * SM_COST_SCALED)/(exp(B_COST  * TRAIN_COST_SCALED) +
                                                            exp(B_COST  * SM_COST_SCALED) + exp(B_COST  * CAR_CO_SCALED))))

V3_GRDM = log( 1 - (1 - exp(B_TIME  * CAR_TT_SCALED)/(exp(B_TIME  * TRAIN_TT_SCALED) +
                                                   exp(B_TIME  * SM_TT_SCALED) + exp(B_TIME  * CAR_TT_SCALED))) *
         (1 - exp(B_COST  * CAR_CO_SCALED)/(exp(B_COST  * TRAIN_COST_SCALED) +
                                                            exp(B_COST  * SM_COST_SCALED) + exp(B_COST  * CAR_CO_SCALED))))

ProbMNL = exp(mu)/(1+exp(mu))
ProbGRDM = 1 - ProbMNL

V_MNL = {1: V1_MNL, 2: V2_MNL, 3: V3_MNL}

V_GRDM = {1: V1_GRDM, 2: V2_GRDM, 3: V3_GRDM}

prob_MNL = models.logit(V_MNL, av, CHOICE)
prob_GRDM = models.logit(V_GRDM, av, CHOICE)
prob = ProbMNL*prob_MNL + ProbGRDM*prob_GRDM
logprob = log(prob)

# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'MNL+RDM'
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False

# %%
# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood(av)

# %%
# Estimate the parameters
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = results.getEstimatedParameters()
pandas_results