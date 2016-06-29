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
# USAGE EXAMPLE -- EQUAL WEIGHT PORTFOLIO:
# 1) if 'exclude_non_overlapping=True' below, the portfolio will only contains 
# days which are available across all of the algo return timeseries.
# if 'exclude_non_overlapping=False' then the portfolio returned will span from the
# earliest startdate of any algo, thru the latest enddate of any algo.
#
# 2) Weight of each algo will always be 1/N where N is the total number of algos passed to the function

portfolio_rets_equal_weight = portfolio_of_algos(algo_rets, exclude_non_overlapping=True)
"""

"""
# USAGE EXAMPLE -- VOLATILITY WEIGHT PORTFOLIO:
# 1) if 'exclude_non_overlapping=True' below, the portfolio will only contains 
# days which are available across all of the algo return timeseries.
# if 'exclude_non_overlapping=False' then the portfolio returned will span from the
# earliest startdate of any algo, thru the latest enddate of any algo.
#
# 2) Weight of each algo will be based on inverse volatility weight. E.g. lower volatility assets get higher weights

def portfolio_of_algos_volatility_weighted(algo_rets,
                                           max_weight_factor,
                                           exclude_non_overlapping=True):
    
    import pyfolio.timeseries
    
    total_portfolio, data_df = portfolio_returns_metric_weighted(
                                        algo_rets.values(), 
                                        weight_function=pf.timeseries.annual_volatility,
                                        weight_function_window=63,                                 
                                        max_weight_factor=max_weight_factor,
                                        exclude_non_overlapping=exclude_non_overlapping,
                                        inverse_weight=True)
    return total_portfolio, data_df

portfolio_rets_vol_weight, raw_data_df = portfolio_of_algos_volatility_weighted(
                                                algo_rets,
                                                max_weight_factor=2.0,
                                                exclude_non_overlapping=True)
"""

"""
# USAGE EXAMPLE -- RISK BUDGET WEIGHTED PORTFOLIO:

def portfolio_of_algos_risk_budget_weighted(algo_rets,
                                            max_weight_factor,
                                            risk_budget_vol=0.06,
                                            unconstrained_gross_leverage=True,
                                            volatility_lookback_window=63
                                            portfolio_rebalance_rule='m'
                                            exclude_non_overlapping=True):
    
    import pyfolio.timeseries
    
    def risk_budget_scale_local(returns, window=21*3, vol_target=risk_budget_vol):
        return risk_budget_scale(returns=returns, window=window, vol_target=vol_target)
    
    total_portfolio, data_df = portfolio_returns_metric_weighted_with_unconstrained_leverage(
                                        algo_rets.values(), 
                                        weight_function=risk_budget_scale_local,                               
                                        weight_function_window=volatility_lookback_window,
                                        portfolio_rebalance_rule=portfolio_rebalance_rule,
                                        max_weight_factor=max_weight_factor,
                                        exclude_non_overlapping=exclude_non_overlapping,
                                        unconstrained_gross_leverage=unconstrained_gross_leverage,
                                        inverse_weight=False)
    
    
    return total_portfolio, data_df


"""



def portfolio_returns(holdings_returns, exclude_non_overlapping=True):
    """Generates an equal-weight portfolio.
    Parameters
    ----------
    holdings_returns : list
       List containing each individual holding's daily returns of the
       strategy, noncumulative.
    exclude_non_overlapping : boolean, optional
       If True, timeseries returned will include values only for dates
       available across all holdings_returns timeseries If False, 0%
       returns will be assumed for a holding until it has valid data
    Returns
    -------
    pd.Series
        Equal-weight returns timeseries.
    """
    port = holdings_returns[0]
    for i in range(1, len(holdings_returns)):
        port = port + holdings_returns[i]

    if exclude_non_overlapping:
        port = port.dropna()
    else:
        port = port.fillna(0)

    return port / len(holdings_returns)


def portfolio_of_algos(algo_rets, exclude_non_overlapping=True):
    import pyfolio.timeseries
    
    total_portfolio = portfolio_returns(algo_rets.values(), exclude_non_overlapping=exclude_non_overlapping)
    return total_portfolio


def min_max_scale_weights(raw_arr, max_weight_factor=2.0):
    min_max_ratio = max(raw_arr) / min(raw_arr)
    
    if min_max_ratio <= 0 or min_max_ratio > max_weight_factor:
        min_max = sklearn.preprocessing.MinMaxScaler(feature_range=(1, max_weight_factor), copy=True)
        temp_re = raw_arr.reshape(-1,1)
        return min_max.fit_transform(temp_re).flatten()
    else:
        return raw_arr


def risk_budget_scale(returns, window=21*3, vol_target=0.06):
    rets = returns[-window:]
    ann_vol = pf.timeseries.annual_volatility(rets)
    scale_factor = vol_target / ann_vol
    
    return scale_factor
    

def portfolio_returns_metric_weighted(holdings_returns,
                                      exclude_non_overlapping=True,
                                      weight_function=None,
                                      weight_function_window=None,
                                      inverse_weight=False,
                                      portfolio_rebalance_rule='q',
                                      max_weight_factor=None
                                      ):
    """
    Generates an equal-weight portfolio, or portfolio weighted by
    weight_function
    Parameters
    ----------
    holdings_returns : list
       List containing each individual holding's daily returns of the
       strategy, noncumulative.
    exclude_non_overlapping : boolean, optional
       (Only applicable if equal-weight portfolio, e.g. weight_function=None)
       If True, timeseries returned will include values only for dates
       available across all holdings_returns timeseries If False, 0%
       returns will be assumed for a holding until it has valid data
    weight_function : function, optional
       Function to be applied to holdings_returns timeseries
    weight_function_window : int, optional
       Rolling window over which weight_function will use as its input values
    inverse_weight : boolean, optional
       If True, high values returned from weight_function will result in lower
       weight for that holding
    portfolio_rebalance_rule : string, optional
       A pandas.resample valid rule. Specifies how frequently to compute
       the weighting criteria
    max_weight_factor : float, optional
       Max amount that one any algo's weight can be relative to any other, based on the weight_function.
       E.g. Value of 2 means no algo can have a weight more than 2x another algo.
    
    Returns
    -------
    (pd.Series, pd.DataFrame)
        pd.Series : Portfolio returns timeseries.
        pd.DataFrame : All the raw data used in the portfolio returns
           calculations
    """

    if weight_function is None:
        if exclude_non_overlapping:
            holdings_df = pd.DataFrame(holdings_returns).T.dropna()
        else:
            holdings_df = pd.DataFrame(holdings_returns).T.fillna(0)

        holdings_df['port_ret'] = holdings_df.sum(axis=1)/len(holdings_returns)
    else:
        holdings_df_na = pd.DataFrame(holdings_returns).T
        holdings_cols = holdings_df_na.columns
        holdings_df = holdings_df_na.dropna()
        holdings_func = pd.rolling_apply(holdings_df,
                                         window=weight_function_window,
                                         func=weight_function).dropna()
        holdings_func_rebal = holdings_func.resample(
            rule=portfolio_rebalance_rule,
            how='last')
        holdings_df = holdings_df.join(
            holdings_func_rebal, rsuffix='_f').fillna(method='ffill').dropna()
        
        if max_weight_factor is not None:
            func_cols = list(map(lambda s: s.endswith('_f'), holdings_df.columns))
            holdings_df_f = holdings_df.ix[:,func_cols]
            
            scaled_rows = [ min_max_scale_weights(ir[1].values, max_weight_factor=max_weight_factor) 
                               for ir in list(holdings_df_f.iterrows()) ]
            scaled_rows_df = pd.DataFrame(scaled_rows)
            
            columns_t = list(map(lambda x: x+"_t",  holdings_cols))
            scaled_rows_df.columns = columns_t        
            for col_idx in scaled_rows_df.columns.values:
                holdings_df[col_idx] = scaled_rows_df[col_idx].values
        else:
            holdings_func_rebal_t = holdings_func_rebal
            holdings_df = holdings_df.join( holdings_func_rebal_t,
                                            rsuffix='_t').fillna(method='ffill').dropna()

        transform_columns = list(map(lambda x: x+"_t", holdings_cols))
        
        if inverse_weight:
            print "Applying inverse weight for metric..."
            inv_func = 1.0 / holdings_df[transform_columns]
            holdings_df_weights = inv_func.div(inv_func.sum(axis=1),
                                               axis='index')
        else:
            holdings_df_weights = holdings_df[transform_columns] \
                .div(holdings_df[transform_columns].sum(axis=1), 
                     axis='index')

        holdings_df_weights.columns = holdings_cols
        holdings_df = holdings_df.join(holdings_df_weights, rsuffix='_w')
        holdings_df_weighted_rets = np.multiply(
            holdings_df[holdings_cols], holdings_df_weights)
        holdings_df_weighted_rets['port_ret'] = holdings_df_weighted_rets.sum(
            axis=1)
        holdings_df = holdings_df.join(holdings_df_weighted_rets,
                                       rsuffix='_wret')

    return holdings_df['port_ret'], holdings_df


def portfolio_returns_metric_weighted_with_unconstrained_leverage(holdings_returns,
                                      exclude_non_overlapping=True,
                                      weight_function=None,
                                      weight_function_window=None,
                                      inverse_weight=False,
                                      portfolio_rebalance_rule='q',
                                      max_weight_factor=None,
                                      unconstrained_gross_leverage=True
                                      ):
    """
    Generates an equal-weight portfolio, or portfolio weighted by
    weight_function
    Parameters
    ----------
    holdings_returns : list
       List containing each individual holding's daily returns of the
       strategy, noncumulative.
    exclude_non_overlapping : boolean, optional
       (Only applicable if equal-weight portfolio, e.g. weight_function=None)
       If True, timeseries returned will include values only for dates
       available across all holdings_returns timeseries If False, 0%
       returns will be assumed for a holding until it has valid data
    weight_function : function, optional
       Function to be applied to holdings_returns timeseries
    weight_function_window : int, optional
       Rolling window over which weight_function will use as its input values
    inverse_weight : boolean, optional
       If True, high values returned from weight_function will result in lower
       weight for that holding
    portfolio_rebalance_rule : string, optional
       A pandas.resample valid rule. Specifies how frequently to compute
       the weighting criteria
    max_weight_factor : float, optional
       Max amount that one any algo's weight can be relative to any other, based on the weight_function.
       E.g. Value of 2 means no algo can have a weight more than 2x another algo.
       If None, then weights returned will be unconstrained.
    unconstrained_gross_leverage : float, optional
       If True, then sum of portfolio weights can be >100% or <100%.  
       If False, weights returned will always sum to 100%.
    
    Returns
    -------
    (pd.Series, pd.DataFrame)
        pd.Series : Portfolio returns timeseries.
        pd.DataFrame : All the raw data used in the portfolio returns
           calculations
    """

    if weight_function is None:
        if exclude_non_overlapping:
            holdings_df = pd.DataFrame(holdings_returns).T.dropna()
        else:
            holdings_df = pd.DataFrame(holdings_returns).T.fillna(0)

        holdings_df['port_ret'] = holdings_df.sum(axis=1)/len(holdings_returns)
    else:
        holdings_df_na = pd.DataFrame(holdings_returns).T
        holdings_cols = holdings_df_na.columns
        holdings_df = holdings_df_na.dropna()
        holdings_func = pd.rolling_apply(holdings_df,
                                         window=weight_function_window,
                                         func=weight_function).dropna()
        holdings_func_rebal = holdings_func.resample(
            rule=portfolio_rebalance_rule,
            how='last')
        holdings_df = holdings_df.join(
            holdings_func_rebal, rsuffix='_f').fillna(method='ffill').dropna()
        
        if max_weight_factor is not None:
            func_cols = list(map(lambda s: s.endswith('_f'), holdings_df.columns))
            holdings_df_f = holdings_df.ix[:,func_cols]
            
            scaled_rows = [ min_max_scale_weights(ir[1].values, max_weight_factor=max_weight_factor) 
                               for ir in list(holdings_df_f.iterrows()) ]
            scaled_rows_df = pd.DataFrame(scaled_rows)
            
            columns_t = list(map(lambda x: x+"_t",  holdings_cols))
            scaled_rows_df.columns = columns_t        
            for col_idx in scaled_rows_df.columns.values:
                holdings_df[col_idx] = scaled_rows_df[col_idx].values
        else:
            holdings_func_rebal_t = holdings_func_rebal
            holdings_df = holdings_df.join( holdings_func_rebal_t,
                                            rsuffix='_t').fillna(method='ffill').dropna()

        transform_columns = list(map(lambda x: x+"_t", holdings_cols))
        
        if unconstrained_gross_leverage:
            print "Returning weights with *unconstrained* leverage. E.g.: Sum of weights can be >100% ..."
                
        if inverse_weight:
            print "Applying inverse weight for metric..."
            inv_func = 1.0 / holdings_df[transform_columns]
            
            if unconstrained_gross_leverage:
                holdings_df_weights = inv_func / len(holdings_returns)
            else:
                holdings_df_weights = inv_func.div(inv_func.sum(axis=1), axis='index')
        else:
            if unconstrained_gross_leverage:
                holdings_df_weights = holdings_df[transform_columns] / len(holdings_returns)
            else:
                holdings_df_weights = holdings_df[transform_columns].div(holdings_df[transform_columns].sum(axis=1), 
                                                                         axis='index')

        holdings_df_weights.columns = holdings_cols
        holdings_df = holdings_df.join(holdings_df_weights, rsuffix='_w')
        
        weights_columns = list(map(lambda x: x+"_w", holdings_cols))
        holdings_df['weights_sum'] = holdings_df[weights_columns].sum(axis=1)
        
        holdings_df_weighted_rets = np.multiply(
            holdings_df[holdings_cols], holdings_df_weights)
        holdings_df_weighted_rets['port_ret'] = holdings_df_weighted_rets.sum(
            axis=1)
        holdings_df = holdings_df.join(holdings_df_weighted_rets,
                                       rsuffix='_wret')
        
        print "Using a rolling window of " + str(weight_function_window) + " trading days for the calculations. "
        print "Rebalance frequency = " + portfolio_rebalance_rule
        print "Max weight constraint of no algo being more than Nx versus any other = " + str(max_weight_factor)
        
    return holdings_df['port_ret'], holdings_df
