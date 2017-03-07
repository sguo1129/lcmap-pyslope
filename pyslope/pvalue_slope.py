"""Functions and classes used by the change detection procedures

This allows for a layer of abstraction, where the individual loop procedures
can be looked at from a higher level
"""

import scipy
import numpy as np
from scipy.stats import f, linregress
import math

from pyslope import app
from ccd.math_utils import euclidean_norm_sq

log = app.logging.getLogger(__name__)
defaults = app.defaults

def update_processing_mask(mask, index, window=None):
    """
    Update the persistent processing mask.

    Because processes apply the mask first, index values given are in relation
    to that. So we must apply the mask to itself, then update the boolean
    values.

    The window slice object is to catch when it is in relation to some
    window of the masked values. So, we must mask against itself, then look at
    a subset of that result.

    This method should create a new view object to avoid mutability issues.

    Args:
        mask: 1-d boolean ndarray, current mask being used
        index: int/list/tuple of index(es) to be excluded from processing,
            or boolean array
        window: slice object identifying a further subset of the mask

    Returns:
        1-d boolean ndarray
    """
    m = mask[:]
    sub = m[m]

    if window:
        sub[window][index] = False
    else:
        sub[index] = False

    m[m] = sub

    return m

def is_leap_year(year):
    """ if year is a leap year return True
        else return False """
    if year % 100 == 0:
        return year % 400 == 0
    return year % 4 == 0

def julian_day_to_doy(jd):
    """
    Convert Julian Day to year and day of year (doy)
    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
        4th ed., Duffet-Smith and Zwart, 2011.
    Parameters
    ----------
    jd : float
        Julian Day

    Returns
    doy : int
        Day of Year
    """
    jd = jd + 0.5
    F, I = math.modf(jd)
    I = int(I)
    A = math.trunc((I - 1867216.25) / 36524.25)
    if I > 2299160:
        B = I + 1 + A - math.trunc(A / 4.)
    else:
        B = I
    C = B + 1524
    D = math.trunc((C - 122.1) / 365.25)
    E = math.trunc(365.25 * D)
    G = math.trunc((C - E) / 30.6001)
    day = C - E + F - math.trunc(30.6001 * G)
    if G < 13.5:
        month = G - 1
    else:
        month = G - 13
    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715

    if is_leap_year(year):
        K = 1
    else:
        K = 2
    doy = int((275 * month) / 9.0) - K * int((month + 9) / 12.0) + day - 30

    return int(doy)


def reg_fast(dates, obs, nobs):
    """Calculate pvalues and slope/constant for the growing season NDVI, EVI, and NBR

    Args:
        dates: 1-d ndarray of ordinal day values
        obs: 1d ndarray representing the one spectral band values
        nobs: number of observations used

    Returns:
        p_vale: static values for
        slope: slope for linear regression
        intercept: intercept for linear regression
    """
    slope, intercept, r_value, p_value, std_err = linregress(dates, obs)

    if (nobs > 2):
        nu = nobs -2
    else:
        nu = 0

    y_hat = slope * dates + intercept
    sum_y = np.sum(obs)
    norm_r = np.zeros(nobs, dtype=float)
    norm_r = [euclidean_norm_sq(obs[i]-y_hat[i])
                  for i in range(nobs)]
    sum_norm_r = np.sum(norm_r)
    mean_y = sum_y / nobs
    rmse = math.sqrt(sum_norm_r)/ math.sqrt(nobs)

    norm_y = np.zeros(nobs, dtype=float)
    norm_y = [euclidean_norm_sq(y_hat[i]-mean_y)
                  for i in range(nobs)]
    sum_norm_y = np.sum(norm_y)
    x_under = (sum_norm_y / (rmse * rmse))

    p_value = f.cdf(x_under, nu, 1)

    return p_value, slope, intercept

def calc_pvalue_slope(dates, observations, processing_mask, start_days, end_days,
                      grow_start=defaults.GROW_START_DAY, grow_end=defaults.GROW_END_DAY):
    """Calculate pvalues and slope/constant for the growing season NDVI, EVI, and NBR
      
    Args:
        dates: 1-d ndarray of ordinal day values
        observations: 2-d ndarray representing the spectral values
        processing_mask: 1-d boolean array identifying which values to
                    consider for processing
        result: Change models for each observation of each spectra.
        grow_start: growing season start day of year
        grow_end: growing season end day of year

    Returns:
        pvalue_slope: pavlus, slope, and intercept for NDVI, EVI, NBR
    """
    _, sample_count = observations.shape
    no_grow_index = np.zeros(sample_count, dtype=bool)

    pvalue_slope=[]

    for i in range(sample_count):
        doy = julian_day_to_doy(dates[i])
        if (doy < grow_start or doy > grow_end):
            no_grow_index[i] = False

    processing_mask = update_processing_mask(processing_mask, no_grow_index)

    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]
    nperiod = len(period)

    ndvi = np.zeros(nperiod, dtype=float)
    evi = np.zeros(nperiod, dtype=float)
    nbr = np.zeros(nperiod, dtype=float)

    ndvi = (spectral_obs[3, :] - spectral_obs[2, :]) \
               /(spectral_obs[3, :] - spectral_obs[2, :] + 0.01)
    evi = 2.5 * (spectral_obs[3, :] - spectral_obs[2, :]) \
               /(spectral_obs[3, :] + 6.0 * spectral_obs[2, :] \
                 - 7.5 * spectral_obs[0, :] + 10000.0)
    nbr = (spectral_obs[3, :] - spectral_obs[5, :]) \
               / (spectral_obs[3, :] - spectral_obs[5, :] + 0.01)

    temp_mask = np.zeros(nperiod, dtype=bool)
#    for num, results in enumerate(results['change_model']):
    for i in range(len(start_days)):
        for j in range(sample_count):
            if (period[j] < start_days[i] and \
                            period[j] > end_days[i]):
                no_grow_index[j] = False
        temp_mask = update_processing_mask(processing_mask, no_grow_index)
        cur_period = period[temp_mask]
        ndvi = ndvi[temp_mask]
        evi = evi[temp_mask]
        nbr = nbr[temp_mask]
        pvalue_ndvi, slope_ndvi, intercept_ndvi = \
            reg_fast(cur_period, ndvi, len(cur_period))
        pvalue_evi, slope_evi, intercept_evi  = \
            reg_fast(cur_period, evi, len(cur_period))
        pvalue_nbr, slope_nbr, intercept_nbr  = \
            reg_fast(cur_period, nbr, len(cur_period))

        pslope = {'ndvi_pvalue': float,
                        'ndvi_slope': float,
                        'ndvi_intercept': float,
                        'evi_pvalue': float,
                        'evi_slope': float,
                        'evi_intercept': float,
                        'nbr_pvalue': float,
                        'nbr_slope': float,
                        'nbr_intercept': float
                        }

        pslope['ndvi_pvalue'] = pvalue_ndvi
        pslope['ndvi_slope'] = slope_ndvi
        pslope['ndvi_intercept'] = intercept_ndvi
        pslope['evi_pvalue'] = pvalue_evi
        pslope['evi_slope'] = slope_evi
        pslope['evi_intercept'] = intercept_evi
        pslope['nbr_pvalue'] = pvalue_nbr
        pslope['nbr_slope'] = slope_nbr
        pslope['nbr_intercept'] = intercept_nbr
        pvalue_slope.append(pslope)

        # print(pvalue_slope[i])
        print('i=',i)
        print(pvalue_slope[i]['ndvi_slope'])
        print(pvalue_slope[i]['ndvi_slope'])
        print(pvalue_slope[i]['ndvi_intercept'])
        print(pvalue_slope[i]['evi_pvalue'])
        print(pvalue_slope[i]['evi_slope'])
        print(pvalue_slope[i]['evi_intercept'])
        print(pvalue_slope[i]['nbr_pvalue'])
        print(pvalue_slope[i]['nbr_slope'])
        print(pvalue_slope[i]['nbr_intercept'])

    """
        result = results_to_pvalue_slope(
                                        pvalue[0]=pvalue_ndvi,
                                        pvalue[1]=pvalue_evi,
                                        pvalue[2]=pvalue_nbr
                                        slope[0]=slope_ndvi,
                                        slope[1]=slope_evi,
                                        slope[2]=slope_nbr,
                                        const[0]=intercept_ndvi,
                                        const[1]=intercept_evi,
                                        const[2]=intercept_nbr)
    """
    return pvalue_slope