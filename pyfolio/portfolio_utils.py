#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import division

from collections import OrderedDict
from functools import partial

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats

from . import utils
from .utils import APPROX_BDAYS_PER_MONTH, APPROX_BDAYS_PER_YEAR
from .utils import DAILY, WEEKLY, MONTHLY, YEARLY, ANNUALIZATION_FACTORS
from .interesting_periods import PERIODS

"""
# USAGE EXAMPLE -- CONE POSITION WEIGHTED PORTFOLIO:
# 1) if 'exclude_non_overlapping=True' below, the portfolio will only contains 
# days which are available across all of the algo return timeseries.
# if 'exclude_non_overlapping=False' then the portfolio returned will span from the
# earliest startdate of any algo, thru the latest enddate of any algo.
# 
# Weight of holding will be based on how many z-scores away from its mean it is currently

def portfolio_of_algos_cone_weighted(algo_rets, max_weight_factor, exclude_non_overlapping=True):
    
    import pyfolio.timeseries
    
    total_portfolio, data_df = portfolio_returns_metric_weighted(
                                        algo_rets.values(), 
                                        max_weight_factor=max_weight_factor,
                                        weight_function=cone_z_score,
                                        weight_function_window=21*13, 
                                        inverse_weight=True
                                        )
    return total_portfolio, data_df

portfolio_rets_cone_weight, raw_data_df = portfolio_of_algos_cone_weighted(
                                                        algo_rets,
                                                        max_weight_factor=2.0,
                                                        exclude_non_overlapping=True)
                                                        
"""

def cone_z_score(returns, short_window=21*6, long_window=21*12*5, abs_value=True):
    
    long_window_rets = returns[-long_window:-short_window]

    daily_mean_long_window = np.mean( long_window_rets )
    daily_std_long_window = np.std( long_window_rets)
    
    daily_mean_short_window = np.mean( returns[-short_window:] )
    algo_ret = (1.0 + daily_mean_short_window)**short_window - 1.0
    

    
    expected_ret = (1.0 + daily_mean_long_window)**short_window - 1.0
    expected_std = daily_std_long_window * np.sqrt(short_window)
    
    cone_z = (algo_ret - expected_ret) / expected_std
    
    if abs_value:
        cone_z = np.abs(cone_z)
        
    return cone_z
    
"""
# USAGE EXAMPLE -- KELLY WEIGHTED PORTFOLIO:
# 1) if 'exclude_non_overlapping=True' below, the portfolio will only contains 
# days which are available across all of the algo return timeseries.
# if 'exclude_non_overlapping=False' then the portfolio returned will span from the
# earliest startdate of any algo, thru the latest enddate of any algo.
# 
# Weight of holding will be based on kelly ratio

def portfolio_of_algos_kelly_weighted(algo_rets, max_weight_factor, exclude_non_overlapping=True):
    
    import pyfolio.timeseries
    
    total_portfolio, data_df = portfolio_returns_metric_weighted(
                                        algo_rets.values(), 
                                        max_weight_factor=max_weight_factor,
                                        weight_function=kelly_calc,
                                        weight_function_window=21*13, 
                                        inverse_weight=False
                                        )
    return total_portfolio, data_df

portfolio_rets_kelly_weight, raw_data_df = portfolio_of_algos_kelly_weighted(
                                                        algo_rets,
                                                        max_weight_factor=2.0,
                                                        exclude_non_overlapping=True)
"""

def kelly_calc(returns, window=21*12):
    rets = returns[-window:]
    return np.mean(rets) / np.var(rets)
    
# helper function to create discretized buckets for all values in input vector
def bucket_std(value, bins=[0.12, 0.15, 0.18, 0.21], max_default=0.24):
    """
    Simple quantizing function. For use in binning stdevs into a "buckets"
    Parameters
    ----------
    value : float
       Value corresponding to the the stdev to be bucketed
    bins : list, optional
       Floats used to describe the buckets which the value can be placed
    max_default : float, optional
       If value is greater than all the bins, max_default will be returned
    Returns
    -------
    float
        bin which the value falls into
    """

    annual_vol = value * np.sqrt(252)

    for i in bins:
        if annual_vol <= i:
            return i

    return max_default

# helper function to winsorize volatility values to upper and lower bounds
def min_max_vol_bounds(value, lower_bound=0.12, upper_bound=0.24):
    """
    Restrict volatility weighting of the lowest volatility asset versus the
    highest volatility asset to a certain limit.
    E.g. Never allocate more than 2x to the lowest volatility asset.
    round up all the asset volatilities that fall below a certain bound
    to a specified "lower bound" and round down all of the asset
    volatilites that fall above a certain bound to a specified "upper bound"
    Parameters
    ----------
    value : float
       Value corresponding to a daily volatility
    lower_bound : float, optional
       Lower bound for the volatility
    upper_bound : float, optional
       Upper bound for the volatility
    Returns
    -------
    float
        The value input, annualized, or the lower_bound or upper_bound
    """

    annual_vol = value * np.sqrt(252)

    if annual_vol < lower_bound:
        return lower_bound

    if annual_vol > upper_bound:
        return upper_bound

    return annual_vol
