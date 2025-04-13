# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 17:13:34 2025

@author: d23gr
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import extract_asset
# import get_CPI_Udata 

# stock market financial data
# import yfinance as yf

# consumer price index data
# import requests
# import json
# import prettytable

##################################### Plot Function ###########################
# function to plat data
def plot_lineData(ax, x, y, my_title, xlabel, ylabel, line_color, legend_label, width):
    # Specify color pallate suitable for people with colorblindness
    colorblind_palette = {
            "light_green":  "#CCFF99",
            "dark_green":   "#009900",
            "light_blue":   "#99CCFF",
            "dark_blue":    "#0066CC",
            "light_red":    "#FF9999",
            "dark_red":     "#CC0000"
        }

    # plot parameters
    use_color = colorblind_palette[line_color]
    ax.plot(x, y, 
            color = use_color, 
            label = legend_label,
            linewidth = width)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    plt.title(my_title)
    
def plot_bar(ax, x, y, bar_colors, x_label=None, y_label=None, title=None):
    """
    Plots bar data on a given subplot axis.

    Parameters:
    ax (matplotlib.axes._subplots.AxesSubplot): The subplot axis to plot on.
    x (list or array): Data for the x-axis (e.g., categories).
    y (list or array): Data for the y-axis (e.g., bar heights).
    x_label (str, optional): Label for the x-axis.
    y_label (str, optional): Label for the y-axis.
    title (str, optional): Title of the plot.
    bar_color (str or list, optional): Color of the bars (default is 'blue').

    Returns:
    None
    """
    
    ax.bar(x, y, color=bar_colors)
    ax.set_xlabel(x_label if x_label else '')
    ax.set_ylabel(y_label if y_label else '')
    ax.set_title(title if title else '')
    plt.xticks(rotation=45)  # Rotate labels by 45 degrees

# Calculating max drawdown
def calculate_max_drawdown(prices):
    # Calculate the running maximum
    running_max = prices.cummax()
    # Calculate the drawdown
    drawdown = (prices - running_max) / running_max
    # Find the maximum drawdown
    max_drawdown = drawdown.min()
    return max_drawdown

# Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate):
    risk_free_rate = 0.0001  # Assume daily risk-free rate
    # Excess returns (returns above the risk-free rate)
    excess_returns = returns - risk_free_rate
    # Calculate Sharpe Ratio
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    trading_days = len(returns)
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(trading_days)
    return sharpe_ratio, annualized_sharpe_ratio

################################# Set up ######################################
# set input parameters
today = dt.today()
today_str = today.strftime('%Y-%m-%d')

############################### Asset figure ##############################
# specify terms of interest
term_names =['obama1','obama2','trump1','biden','trump2']
start_terms = ['2009-01-20','2013-01-20','2017-01-20','2021-01-20','2025-01-20']
end_terms = ['2013-01-20','2017-01-20','2021-01-20','2025-01-20',today_str]

# specify asset of interest
assets = ['^GSPC']

# plotting details
x_label = 'Work Days Since Innauguration'
y_label = '% Indexed to Inauguration Day'
title = 'S&P 500 Scaled to First Day in Office'
yrs = 1
days_out = 365*yrs

#################################### Scaled S&P500 ############################
# Extract data for the S&P500
asset_dict = extract_asset.get_asset(days_out, assets, 
                       term_names, start_terms, end_terms
                       )

for asset in asset_dict.keys():
    asset_df = asset_dict[asset]
# plot color list
color_list = ["light_green","dark_green",
            "light_blue",
            "light_red","dark_red"
            ]

# MAIN PLOT OF STOCK VALUES OVER TIME
# Create a figure and axes
fig, ax = plt.subplots()
plt.style.use('tableau-colorblind10')

for i, term_name in enumerate(term_names):
    legend_label = term_name
    x_data = asset_df[asset_df['Term_name'] == term_names[i]]['Date'] - asset_df[asset_df['Term_name'] == term_names[i]]['Date'].iloc[0]
    x_trim = x_data[x_data < f'{days_out} days 00:00:00']
    x_days = x_trim.dt.days.astype(str) + ' days'
    x = [day for day in range(0,len(x_days))]
    y = asset_df[asset_df['Term_name'] == term_names[i]][f'{asset} Indexed'].values
    width = 1
    if i == len(term_names)-1:
        width = 3
    plot_lineData(ax, x, y[:x[-1]+1], 
                title, x_label, y_label, 
                color_list[i], legend_label, width
                )

plt.savefig("economy_S&P500byTerm.png")  # Saves the figure to a .png file
plt.show()

########################## Metrics of interest
######################### Post-pre change in value ############################
# Calculated as the final value of the asset minus the first value
# Create a figure and axes
fig, ax = plt.subplots(1,3)
plt.style.use('tableau-colorblind10')

term_list = []
delta_list = []

for i, term_name in enumerate(term_names):
    legend_label = term_name
    # x_data = asset_df[asset_df['Term_name'] == term_names[i]]['Date'] - asset_df[asset_df['Term_name'] == term_names[i]]['Date'].iloc[0]
    # x_trim = x_data[x_data < f'{days_out} days 00:00:00']
    # x_days = x_trim.dt.days.astype(str) + ' days'
    # x = [day for day in range(0,len(x_days))]
    y = asset_df[asset_df['Term_name'] == term_names[i]][f'{asset} Indexed'].values
    term_list.append(term_name)
    delta_list.append(float(y[-1] - y[0]))

delta_dict = {'terms': term_list,
             'asset delta': delta_list}
diff_df = pd.DataFrame(delta_dict)

x = diff_df['terms']
y = diff_df['asset delta']
# Specify color pallate suitable for people with colorblindness
colors = ["#CCFF99", "#009900",
          "#FF9999",
          "#99CCFF",
          "#CC0000"
          ]

plot_bar(ax[0], x, y,
        colors,
        x_label='Administration', 
        y_label='% Change Since Inauguration', 
        title='Change in S&P500')

plt.show()
plt.savefig("economy_S&P500_pre-post.png")  # Saves the figure to a .png file

########################## Market volatility ##################################
# Calculated as the mean squared error of the residuals from the data linear model

########################## Maximum drawdown ###################################

########################## Sharpe Ratio #######################################

