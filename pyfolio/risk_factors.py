#
# Copyright 2016 Quantopian, Inc.
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

import pandas as pd
import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.lines as mlines

from functools import partial


"""
WARNING:  THIS IS ALL VERY ROUGH.  It works standalone, but needs to be fully 
integrated into the pyfolio framework after a possible re-factoring and 
generalization of how risk-factors are computed and plotted
"""

"""
All of this sector risk factor and macro risk factor work needs
to be better integrated into a more generic framework for operating similarly to 
the Fama-French risk factor analysis.  The Fama-French code was not generalized
enough when developing this sector/macro risk factor analysis, so it should
be refactored into a more generalized risk factor function that accepts arbitrary 
timeseries to serve as risk-factors
"""

SPY_px = web.get_data_yahoo('SPY',start='2001-1-1')
SPY_px = pd.DataFrame.rename(SPY_px, columns={'Adj Close': 'AdjClose'})
SPY_rets = SPY_px.AdjClose.pct_change().dropna()
SPY_rets.name = "SPY"
SPY_rets.index = SPY_rets.index.tz_localize('UTC').normalize()

oil_px = web.get_data_yahoo('USO',start='2001-1-1')
oil_px = pd.DataFrame.rename(oil_px, columns={'Adj Close': 'AdjClose'})
oil_rets = oil_px.AdjClose.pct_change().dropna()
oil_rets.name = "Oil"
oil_rets.index = oil_rets.index.tz_localize('UTC').normalize()

gold_px = web.get_data_yahoo('GLD',start='2001-1-1')
gold_px = pd.DataFrame.rename(gold_px, columns={'Adj Close': 'AdjClose'})
gold_rets = gold_px.AdjClose.pct_change().dropna()
gold_rets.name = "Gold"
gold_rets.index = gold_rets.index.tz_localize('UTC').normalize()

bond_px = web.get_data_yahoo('TLT',start='2001-1-1')
bond_px = pd.DataFrame.rename(bond_px, columns={'Adj Close': 'AdjClose'})
bond_rets = bond_px.AdjClose.pct_change().dropna()
bond_rets.name = "T_Bond_20yr"
bond_rets.index = bond_rets.index.tz_localize('UTC').normalize()

vix_px = web.get_data_yahoo('^VIX',start='2001-1-1')
vix_px = pd.DataFrame.rename(vix_px, columns={'Adj Close': 'AdjClose'})
vix_rets = vix_px.AdjClose.pct_change().dropna()
vix_rets.name = "VIX"
vix_rets.index = vix_rets.index.tz_localize('UTC').normalize()

ir30_px = web.get_data_yahoo('^TYX',start='2001-1-1')
ir30_px = pd.DataFrame.rename(ir30_px, columns={'Adj Close': 'AdjClose'})
ir30_rets = ir30_px.AdjClose.pct_change().dropna()
ir30_rets.name = "30yr_Int_Rate"
ir30_rets.index = ir30_rets.index.tz_localize('UTC').normalize()

ir10_px = web.get_data_yahoo('^TNX',start='2001-1-1')
ir10_px = pd.DataFrame.rename(ir10_px, columns={'Adj Close': 'AdjClose'})
ir10_rets = ir10_px.AdjClose.pct_change().dropna()
ir10_rets.name = "10yr_Int_Rate"
ir10_rets.index = ir10_rets.index.tz_localize('UTC').normalize()

curve30_10_px = ir30_px - ir10_px
curve30_10_rets = curve30_10_px.AdjClose.pct_change().dropna()
curve30_10_rets.name = "Yield_Curve_30_10"
curve30_10_rets.index = curve30_10_rets.index.tz_localize('UTC').normalize()

# Sector ETF's
XLY_px = web.get_data_yahoo('XLY',start='2001-1-1')
XLY_px = pd.DataFrame.rename(XLY_px, columns={'Adj Close': 'AdjClose'})
XLY_rets = XLY_px.AdjClose.pct_change().dropna()
XLY_rets.name = "Consumer_Discretionary_XLY"
XLY_rets.index = XLY_rets.index.tz_localize('UTC').normalize()

XLP_px = web.get_data_yahoo('XLP',start='2001-1-1')
XLP_px = pd.DataFrame.rename(XLP_px, columns={'Adj Close': 'AdjClose'})
XLP_rets = XLP_px.AdjClose.pct_change().dropna()
XLP_rets.name = "Consumer_Staples_XLP"
XLP_rets.index = XLP_rets.index.tz_localize('UTC').normalize()

XLE_px = web.get_data_yahoo('XLE',start='2001-1-1')
XLE_px = pd.DataFrame.rename(XLE_px, columns={'Adj Close': 'AdjClose'})
XLE_rets = XLE_px.AdjClose.pct_change().dropna()
XLE_rets.name = "Energy_XLE"
XLE_rets.index = XLE_rets.index.tz_localize('UTC').normalize()

XLF_px = web.get_data_yahoo('XLF',start='2001-1-1')
XLF_px = pd.DataFrame.rename(XLF_px, columns={'Adj Close': 'AdjClose'})
XLF_rets = XLF_px.AdjClose.pct_change().dropna()
XLF_rets.name = "Financials_XLF"
XLF_rets.index = XLF_rets.index.tz_localize('UTC').normalize()

XLV_px = web.get_data_yahoo('XLV',start='2001-1-1')
XLV_px = pd.DataFrame.rename(XLV_px, columns={'Adj Close': 'AdjClose'})
XLV_rets = XLV_px.AdjClose.pct_change().dropna()
XLV_rets.name = "Health_Care_XLV"
XLV_rets.index = XLV_rets.index.tz_localize('UTC').normalize()

XLI_px = web.get_data_yahoo('XLI',start='2001-1-1')
XLI_px = pd.DataFrame.rename(XLI_px, columns={'Adj Close': 'AdjClose'})
XLI_rets = XLI_px.AdjClose.pct_change().dropna()
XLI_rets.name = "Industrials_XLI"
XLI_rets.index = XLI_rets.index.tz_localize('UTC').normalize()

XLB_px = web.get_data_yahoo('XLB',start='2001-1-1')
XLB_px = pd.DataFrame.rename(XLB_px, columns={'Adj Close': 'AdjClose'})
XLB_rets = XLB_px.AdjClose.pct_change().dropna()
XLB_rets.name = "Materials_XLB"
XLB_rets.index = XLB_rets.index.tz_localize('UTC').normalize()

XLK_px = web.get_data_yahoo('XLK',start='2001-1-1')
XLK_px = pd.DataFrame.rename(XLK_px, columns={'Adj Close': 'AdjClose'})
XLK_rets = XLK_px.AdjClose.pct_change().dropna()
XLK_rets.name = "Tech_XLK"
XLK_rets.index = XLK_rets.index.tz_localize('UTC').normalize()

XLU_px = web.get_data_yahoo('XLU',start='2001-1-1')
XLU_px = pd.DataFrame.rename(XLU_px, columns={'Adj Close': 'AdjClose'})
XLU_rets = XLU_px.AdjClose.pct_change().dropna()
XLU_rets.name = "Utilities_XLU"
XLU_rets.index = XLU_rets.index.tz_localize('UTC').normalize()


factor_rets = [SPY_rets, oil_rets, gold_rets, bond_rets, vix_rets, curve30_10_rets
              ,XLY_rets, XLP_rets, XLE_rets, XLF_rets, XLV_rets, XLI_rets, XLB_rets, XLK_rets, XLU_rets 
              ]
factor_rets_df = pd.DataFrame(factor_rets).T.dropna()

sector_rets = [ XLY_rets, XLP_rets, XLE_rets, XLF_rets, XLV_rets, XLI_rets, XLB_rets, XLK_rets, XLU_rets ]
sector_rets_df = pd.DataFrame(sector_rets).T.dropna()

macro_rets = [ SPY_rets, oil_rets, gold_rets, bond_rets, vix_rets, curve30_10_rets ]
macro_rets_df = pd.DataFrame(macro_rets).T.dropna()

def print_risk_factors(returns, factor_rets_df, lookback_days):
    correl_window = lookback_days

    # first grab some of slices of the most recent trading days of an algo to use later
    ret_slice_tail = returns[-correl_window:].dropna()  

    factor_corr_df = pd.DataFrame( [ (col, ret_slice_tail.corr(ser)) for col,ser in factor_rets_df.iteritems() ] )
    factor_beta_df = pd.DataFrame( [ (col, pd.ols(y=ret_slice_tail, x=ser).beta.x) for col,ser in factor_rets_df.iteritems() ] )
    
    factor_corr_df.columns = ['factor', 'correl']
    factor_beta_df.columns = ['factor', 'beta']
    
    factor_corr_df.index = factor_corr_df.factor
    factor_beta_df.index = factor_beta_df.factor
    
    factor_corr_df['correl_abs'] = factor_corr_df['correl'].abs()
    
    factor_corr_df.drop('factor', 1, inplace=True)
    factor_beta_df.drop('factor', 1, inplace=True)
    
    factor_corr_df = factor_beta_df.join(factor_corr_df)
    factor_corr_df = factor_corr_df.sort('correl_abs', ascending=False)
    factor_corr_df = np.round( factor_corr_df, 2 )
    
    top_risk_factors = ( factor_corr_df.index[0], factor_corr_df.iloc[0].correl, factor_corr_df.iloc[0].beta )
    
    print str(correl_window) + "-trading day lookback window"
    print factor_corr_df.ix[:,['correl','beta']]

def rolling_beta(returns, factor_returns,
                 rolling_window=21 * 6):
    """Determines the rolling beta of a strategy.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series or pd.DataFrame
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
        If DataFrame is passed, computes rolling beta for each column.
    rolling_window : int, optional
        The size of the rolling window, in days, over which to compute
        beta (default 6 months).
    Returns
    -------
    pd.Series
        Rolling beta.
    Note
    -----
    See https://en.wikipedia.org/wiki/Beta_(finance) for more details.
    """
    if factor_returns.ndim > 1:
        # Apply column-wise
        return factor_returns.apply(partial(rolling_beta, returns),
                                    rolling_window=rolling_window)
    else:
        out = pd.Series(index=returns.index)
        for beg, end in zip(returns.index[0:-rolling_window],
                            returns.index[rolling_window:]):
            out.loc[end] = pf.timeseries.alpha_beta(
                returns.loc[beg:end],
                factor_returns.loc[beg:end])[1]

        return out
    
    
def plot_rolling_risk_factor(
        returns,
        factor_returns=None,
        rolling_window=21 * 6,
        legend_loc='best',
        remove_benchmark_beta_from_factor=False,
        benchmark_returns=None,
        ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    ax.set_title(
        "Rolling Single Factor Betas (%.0f-month)" % (
            rolling_window / 21
        )
    )

    ax.set_ylabel('beta')

    if remove_benchmark_beta_from_factor:
        rolling_factor_vs_benchmark_beta = rolling_beta(returns=factor_returns, factor_returns=benchmark_returns, rolling_window=rolling_window)
        daily_factor_alpha_returns = factor_returns - rolling_factor_vs_benchmark_beta * benchmark_returns
        rolling_risk_factor = rolling_beta(returns=returns, factor_returns=daily_factor_alpha_returns, rolling_window=rolling_window)
    else:
        rolling_risk_factor = rolling_beta(returns, factor_returns=factor_returns, rolling_window=rolling_window)
    
    rolling_risk_factor.plot(alpha=0.7, ax=ax, **kwargs)

    ax.axhline(0.0, color='black')
    ax.set_ylim((-0.75, 0.75))

    ax.axhline(0.0, color='black')
    ax.set_xlabel('')

    return ax


def plot_rolling_sector_risk_factors(returns, sector_rets_df):
    factor_rets = sector_rets_df.fillna(0.0)
    cmap = sns.color_palette("Set1", len(factor_rets.columns))

    fig, ax = plt.subplots(figsize=(15, 4))

    rets = returns.dropna()
    rets = rets[ rets.index > factor_rets.index[0] ]
    
    plot_rolling_risk_factor(rets, factor_returns=factor_rets, color=cmap, ax=ax, 
                             title="Sector Risk Factors (Rolling beta)").legend(loc='center left', bbox_to_anchor=(1, 0.5))
    

def plot_rolling_macro_risk_factors(returns, macro_rets_df):
    factor_rets = macro_rets_df.fillna(0.0)
    cmap = sns.color_palette("Set1", len(factor_rets.columns))

    fig, ax = plt.subplots(figsize=(15, 4))

    rets = returns.dropna()
    rets = rets[ rets.index > factor_rets.index[0] ]
    
    plot_rolling_risk_factor(rets, factor_returns=factor_rets, color=cmap, ax=ax, 
                             title="Macro Risk Factors (Rolling beta)").legend(loc='center left', bbox_to_anchor=(1, 0.5))
    

