# -*- coding: utf-8 -*-
'''
Created on Fri Mar 21 17:13:34 2025

@author: d23gr
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import extract_asset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------- Plot Functions ------------------------------------
# function to plat data
def plot_lineData(ax, x, y, my_title, xlabel, ylabel, line_color, legend_label, width):
    # Specify color pallate suitable for people with colorblindness
    colorblind_palette = {
            'light_green':  '#CCFF99',
            'dark_green':   '#009900',
            'light_blue':   '#99CCFF',
            'dark_blue':    '#0066CC',
            'light_red':    '#FF9999',
            'dark_red':     '#CC0000'
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
    
def plot_bar(ax, x, y, bar_colors, x_label=None, y_label=None, title=None, xtickop=None):
    '''
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
    '''
    
    ax.bar(x, y, color=bar_colors)
    ax.set_xlabel(x_label if x_label else '')
    ax.set_ylabel(y_label if y_label else '')
    ax.set_title(title if title else '', fontsize=10)
    ax.tick_params(axis='x', rotation=45)  # Rotate labels by 45 degrees
    if  not xtickop:
        ax.set_xticks([])

# --------------------------------- Data metric funcitons ---------------------
# MSE from linear fit
def fit_lm(X, y):
    
    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Fit the model with training data
    model.fit(X, y)
    
    # Predict the target variable using the test set
    y_pred = model.predict(X)
    
    # Calculate residuals (difference between actual and predicted values)
    # residuals = y - y_pred
    
    # Evaluate the model's performance
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    # r2 = r2_score(y, y_pred)
    
    return rmse, y_pred
    
# Calculating max drawdown
def calculate_mdraw_listdown(prices):
    # Calculate the running maximum
    # running_max = prices.cummax()
    running_max = np.maximum.accumulate(prices)
    
    # Calculate the drawdown
    drawdown = (prices - running_max) / running_max
    # Find the maximum drawdown
    mdraw_listdown = 100 * drawdown.min()
    return mdraw_listdown

# Sharpe Ratio
def calculate_sharpe_ratio(prices, days):
    # Convert prices to returns
    price_series = pd.Series(prices)
    daily_returns = price_series.pct_change().dropna()
    # daily_returns = prices[1:] / prices[:-1] - 1
    
    annual_risk_free_rate = 0.03
    
    # convert annual risk-free rate to daily
    daily_rate = (1 + annual_risk_free_rate) ** (1/252) - 1
    
    span_risk_free_rate = (1 + daily_rate) ** days - 1
    
    # Excess returns (returns above the risk-free rate)
    excess_returns = daily_returns - span_risk_free_rate
    
    # Calculate Sharpe Ratio
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    
    # trading_days = len(prices)
    # annualized_sharpe_ratio = sharpe_ratio * np.sqrt(trading_days)
    return sharpe_ratio

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
yrs = 100/365
days_out = 365*yrs

#################################### Scaled S&P500 ############################
# Extract data for the S&P500
asset_dict = extract_asset.get_asset(days_out, assets, 
                       term_names, start_terms, end_terms
                       )

for asset in asset_dict.keys():
    asset_df = asset_dict[asset]
# plot color list
color_list = ['light_green','dark_green',
            'light_red',
            'light_blue','dark_red'
            ]

# Scaled S&P500 line data
fig1, ax1 = plt.subplots()
plt.style.use('tableau-colorblind10')

# initialize variables for subplot analysis
term_list   = []
delta_list  = []
rmse_list   = []
mdraw_list  = []
shrp_list   = []

for i, term_name in enumerate(term_names):
    legend_label = term_name
    x_data = asset_df[asset_df['Term_name'] == term_names[i]]['Date'] - asset_df[asset_df['Term_name'] == term_names[i]]['Date'].iloc[0]
    x_line = x_data[x_data < f'{days_out} days 00:00:00']
    x_days = x_line.dt.days.astype(str) + ' days'
    x_trim = [day for day in range(0,len(x_days))]
    y_line = asset_df[asset_df['Term_name'] == term_names[i]][f'{asset} Indexed'].values
    y_trim = y_line[:x_trim[-1]+1]
    
    # ---------------------------- Scaled S&P500 ------------------------------
    width = 1
    if i == len(term_names)-1:
        width = 2
    plot_lineData(ax1, x_trim, y_trim, 
                title, x_label, y_label, 
                color_list[i], legend_label, width
                )

    # ---------------------------- Pre-Post S&P500 ----------------------------
    # y_full trimmed to the 'days_out' index
    X = np.array(x_trim).reshape(-1, 1)
    rmse, y_pred = fit_lm(X, y_trim)
    mdraw_listdown = calculate_mdraw_listdown(y_trim)
    sharpe_ratio = calculate_sharpe_ratio(y_trim, days_out)
    
    term_list.append(term_name)
    delta_list.append(float(y_pred[-1] - y_pred[0]))
    rmse_list.append(float(rmse))
    mdraw_list.append(float(mdraw_listdown))
    shrp_list.append(float(sharpe_ratio))


# Pre-post change from linear model
fig2, ax2 = plt.subplots(2,2)
plt.style.use('tableau-colorblind10')

# Pre-Post S&P500 from linear model
delta_dict  = {'terms':          term_list,
               'asset delta':     delta_list,
               'rmse':            rmse_list,
               'max draw':        mdraw_list,
               'sharpe ratio':    shrp_list
               }

analysis_df = pd.DataFrame(delta_dict)

x = analysis_df['terms']
y_scaled    = analysis_df['asset delta']

# Specify color pallate suitable for people with colorblindness
colors      = ['#CCFF99', '#009900',
              '#FF9999',
              '#99CCFF',
              '#CC0000'
              ]
xlabel      = ''

# Plot Pre-post S&P500
plot_bar(ax2[0,0], x, y_scaled,
        colors,
        x_label=xlabel, 
        y_label='% Change', 
        title='Change in S&P500')

# Plot marker volatility (rmse)
y_vol = analysis_df['rmse']
xlabel      = ''

plot_bar(ax2[0,1], x, y_vol,
         colors,
         x_label=xlabel,
         y_label='Volatility (rmse)',
         title='Market Volatility')

# Plot max drawdown
y_maxdr     = analysis_df['max draw']
xlabel      = 'Administration'

plot_bar(ax2[1,0], x, y_maxdr,
         colors,
         x_label=xlabel,
         y_label='% drawdown',
         title='Maximum Drawdown',
         xtickop=True)

# Plot Sharpe ration
y_sharpe   = analysis_df['sharpe ratio']
xlabel     = 'Administration'

plot_bar(ax2[1,1], x, y_sharpe,
         colors,
         x_label=xlabel,
         y_label='Ratio',
         title='Sharpe Ratio',
         xtickop=True)
# ---------------------------- Show and save figs -----------------------------
# Scaled S&P500 
# fig1.show()
fig1.savefig('economy_S&P500byTerm.png')  # Saves the figure to a .png file

# Pre-post S&P500 from linear model
# fig2.show()
fig2.savefig('economy_S&P500_pre-post.png')  # Saves the figure to a .png file

########################## Metrics of interest
######################### Post-pre change in value ############################
# Calculated as the final value of the asset minus the first value
# Create a figure and axes
# fig2, ax2 = plt.subplots(1,3)
# plt.style.use('tableau-colorblind10')

# term_list = []
# delta_list = []

# for i, term_name in enumerate(term_names):
#     legend_label = term_name
#     x_data = asset_df[asset_df['Term_name'] == term_names[i]]['Date'] - asset_df[asset_df['Term_name'] == term_names[i]]['Date'].iloc[0]
#     x_trim = x_data[x_data < f'{days_out} days 00:00:00']
#     x_days = x_trim.dt.days.astype(str) + ' days'
#     # Using the 'x' values allows the trimming of the 'y' data such that the 
#     # end of the period intended is used as the 'last day'
#     x = [day for day in range(0,len(x_days))]
#     y_full = asset_df[asset_df['Term_name'] == term_names[i]][f'{asset} Indexed'].values
#     # y_full trimmed to the 'days_out' index
#     y = y_full[:x[-1]+1]
#     term_list.append(term_name)
#     delta_list.append(float(y[-1] - y[0]))

# delta_dict = {'terms': term_list,
#              'asset delta': delta_list}
# analysis_df = pd.DataFrame(delta_dict)

# x = analysis_df['terms']
# y = analysis_df['asset delta']
# # Specify color pallate suitable for people with colorblindness
# colors = ['#CCFF99', '#009900',
#           '#FF9999',
#           '#99CCFF',
#           '#CC0000'
#           ]

# plot_bar(ax[0], x, y,
#         colors,
#         x_label='Administration', 
#         y_label='% Change', 
#         title='Change in S&P500')

# plt.show()
# plt.savefig('economy_S&P500_pre-post.png')  # Saves the figure to a .png file

########################## Market volatility ##################################
# Calculated as the mean squared error of the residuals from the data linear model

########################## Maximum drawdown ###################################

########################## Sharpe Ratio #######################################

