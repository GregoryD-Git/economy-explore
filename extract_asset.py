# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 09:15:46 2025

@author: d23gr
"""

import riskfolio as rp
import yfinance as yf
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

'''
Stock data presentation
    - Looking at the closing price of the S&P500 over the course of a presidency 
    can help understand how the market is affected by presidential policy
    - The S&P500 is considered an index market for the general state of the 
    market as a whole
    - Tracking the markets closing costs over time, indexed to inauguration day
    can help to understand how the marker changes over time
    
    *analysis ideas???
    - Determine how presidential political affiliation impacts overall market
    progress: 
        NULL HYPOTHESES
        *H0-political affiliation does not predict overall change in market price
        from beginning to end of a presidential term
        *H1-politial affilitation does not predict the overall rate of change in 
        market price over the course of a presidency
'''
def get_asset(days_out, assets, term_names, start_terms, end_terms):
    # Specify how many days out from inauguration we want to see stock data
    # days_out = 100 # number of work days in a year
    
    # today = datetime.today()
    # today_str = today.strftime('%Y-%m-%d')
    
    # term_names =['trump1','biden','trump2']
    # start_terms = ['2017-01-20',
    #                '2021-01-20',
    #                '2025-01-20']
    # end_terms = ['2021-01-20',
    #              '2025-01-20',
    #              today_str]
    
    # assets = ['^GSPC']
    
    # # Specify color pallate suitable for people with colorblindness
    # colorblind_palette = {
    #     "Blue": "#377eb8",
    #     "Orange": "#ff7f00",
    #     "Green": "#4daf4a",
    #     "Pink": "#f781bf",
    #     "Brown": "#a65628",
    #     "Purple": "#984ea3",
    #     "Gray": "#999999",
    #     "Yellow": "#ffff33",
    #     "Red":	"#d73027",
    #     "Light Blue": "#a6cee3",
    #     "Dark Blue": "#1f78b4",
    #     "Light Red": "#fb9a99",
    #     "Dark Red": "#e31a1c"
    # }
    
    # # function to plat data
    # def plot_column(ax, x, y, xlabel, ylabel, line_color, legend_label):
    #     use_color = colorblind_palette[line_color]
    #     ax.plot(x, y, color = use_color, label = legend_label)
    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel(ylabel)
    #     ax.legend()
    #     ax.grid(True)
    
    # initialize dataframe dictionary
    asset_dict = {}
    
    for asset in assets:
        # initialize dataframe
        asset_df = pd.DataFrame()
        
        # collect and format asset data
        for i, start_term in enumerate(start_terms):
            data = yf.download(asset, start= start_term, end= end_terms[i])
            data = data.loc[:,'Close']
            # add list of whos term it is to the dataframe
            names = [term_names[i] for name in range(0,len(data.index))]
            data['Term_name'] = names
            
            # extract column of S&P500
            # columns_used = ['^GSPC']
            columns_used = [asset]
            data[f'{asset} Indexed'] = (data[columns_used] / data[columns_used].iloc[0]) * 100
            
            # concatenate dataframes row over row
            asset_df = pd.concat([asset_df, data], axis=0)
        
        # reset index so dates are a data column
        asset_df.reset_index(inplace=True)
        
        # aapend asset dataframe to asset dictionary
        asset_dict[asset] = asset_df
    
    return asset_dict
    # # plot color list
    # color_list = ['Light Red','Blue','Red']
    
    # # Create a figure and axes
    # fig, ax = plt.subplots(figsize=(10, 6))
    
    # for i, term_name in enumerate(term_names):
    #     legend_label = term_name
    #     x_label = 'Work Days Since Innauguration'
    #     y_label = '% Indexed to Inauguration Day'
    #     x_data = asset_df[asset_df['Term_name'] == term_names[i]]['Date'] - asset_df[asset_df['Term_name'] == term_names[i]]['Date'].iloc[0]
    #     x_trim = x_data[x_data < f'{days_out} days 00:00:00']
    #     x_days = x_trim.dt.days.astype(str) + ' days'
    #     x = [day for day in range(0,len(x_days))]
    #     y = asset_df[asset_df['Term_name'] == term_names[i]]['GSPC Indexed'].values
    #     plot_column(ax, x, y[:x[-1]+1], x_label, y_label, color_list[i], legend_label)
    # plt.show()
    
if __name__ == "__main__":
    get_asset()