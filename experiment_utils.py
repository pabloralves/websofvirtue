"""
Contains auxiliary functions for experimentation
"""

import numpy as np
from scipy.stats import beta

def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    """
    Gets the intersection of two line segments
    See: https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    """
    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return [px, py]

def getAlphaBeta(mu, sigma):
    """
    Calculates the alga and beta parameters of the Beta distribution
    given a desired mu and sigma.

    See: https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance
    """
    alpha = mu**2 * ((1 - mu) / sigma**2 - 1 / mu)
    beta = alpha * (1 / mu - 1)
    return {"alpha": alpha, "beta": beta}

def get_distribution(mean,variance,atol=1e-6):
    """
    Returns the Beta distribution parameters given a desired mean and variance,
    which will then be used to initialize the level of development of the network agents.
    """

    # Get the values
    distribution_parameters = getAlphaBeta(mu=mean, sigma=np.sqrt(variance))

    # Ensure parameters are allowed
    assert mean >= 0, f"Mean ({mean}) must be in [0,1]"
    assert mean <= 1, f"Mean ({mean}) must be in [0,1]"
    assert variance <= mean*(1-mean), f"Can't define Beta distribution for this mean and variance: variance > mean*(1-mean) -> {variance} > {mean*(1-mean)}"

    # Ensure that the resulting distribution
    # has the desired mean and variance wthin the given tolerance level
    real_mean = beta.mean(distribution_parameters['alpha'], distribution_parameters['beta'], loc=0, scale=1)
    real_variance = beta.var(distribution_parameters['alpha'], distribution_parameters['beta'], loc=0, scale=1)
    assert np.allclose([mean], [real_mean], atol=atol), "Generated beta distribution parameters generate an innacurate mean"
    assert np.allclose([variance], [real_variance], atol=atol), "Generated beta distribution parameters generate an innacurate variance"

    return distribution_parameters

def get_development_matrix(distribution_parameters,nodes):
    return beta.rvs(distribution_parameters['alpha'],distribution_parameters['beta'], size=nodes)


def get_individual_morality(m,m0,m1):
    """
    Gets the individual component of their probability to act morally in an interaction based on their level of development, which is linear:
    When m=0, the agent has a probability m0 of behaving morally,
    which grows linearly until reaching a probability m1 when development is m=1
    """
    return m*(m1-m0) + m0

def get_susceptibility(m,x_max,y_max):
    """
    Gets susceptibility of an agent to their group based on their level of moral development (m), which is nonlinear:
    SUceptibility starts at 0, reaches its peak at x_max, with a value of y_max, (etnocentric)
    and finally declines back to 0 when m=1 (worldcentric)
    """
    if m <= x_max:
        # First half of the triangle
        return (y_max/x_max)*m
    elif m > x_max:
        # Second half of the triangle
        return -y_max/(1-x_max)*(m-x_max) + y_max

def get_society_bias(av_m,Sx,Sy):
    """
    Gets social bias term of pairwise interaction, which is nonlinear:
    For an average morality in society under 0.25, the bias remains -max_delta_morality
    For an average morality in society above 0.75, the bias remains +max_delta_morality
    For average moralities in between, it interpolates linearly
    """
    if av_m <= Sx:
        return -Sy
    elif av_m > (1-Sx):
        return Sy
    else:
        return (2*Sy)/(1-2*Sx)*(av_m - Sx) - Sy

def get_agent_stage(m_agent,M12,M23):
    """
    Returns the stage of the agent based on its individual level of moral development m
    0 - Egocentric
    1 - Etchnocentric
    2 - Worldcentric
    """
    if m_agent <= M12:
        return 0
    elif m_agent <= M23:
        return 1
    else:
        return 2