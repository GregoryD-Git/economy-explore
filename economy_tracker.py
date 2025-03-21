# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 09:15:46 2025

@author: d23gr
"""

import riskfolio as rp
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

print('this is a new change')

days_out = 100

colorblind_palette = {
    "Blue": "#377eb8",
    "Orange": "#ff7f00",
    "Green": "#4daf4a",
    "Pink": "#f781bf",
    "Brown": "#a65628",
    "Purple": "#984ea3",
    "Gray": "#999999",
    "Yellow": "#ffff33",
    "Red":	"#d73027",
    "Light Blue": "#a6cee3",
    "Dark Blue": "#1f78b4",
    "Light Red": "#fb9a99",
    "Dark Red": "#e31a1c"
}

today = datetime.today()
today_str = today.strftime('%Y-%m-%d')

term_names =['trump1','biden','trump2']
start_terms = ['2017-01-20',
               '2021-01-20',
               '2025-01-20']
end_terms = ['2021-01-20',
             '2025-01-20',
             today_str]

assets = ['MSFT',
          'DIS',
          '^GSPC'
          ]

# function to plat data
def plot_column(ax, x, y, xlabel, ylabel, line_color, legend_label):
    use_color = colorblind_palette[line_color]
    ax.plot(x, y, color = use_color, label = legend_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

# initialize dataframe
asset_df = pd.DataFrame()

# collect and format
for i, start_term in enumerate(start_terms):
    data = yf.download(assets, start = start_term, end = end_terms[i])
    data = data.loc[:,'Close']
    # add list of whos term it is to the dataframe
    names = [term_names[i] for name in range(0,len(data.index))]
    data['Term_name'] = names
    
    # extract column of S&P500
    columns_used = ['^GSPC']
    data['GSPC Indexed'] = (data[columns_used] / data[columns_used].iloc[0]) * 100
    
    asset_df = pd.concat([asset_df, data], axis=0)

# get days out from inauguration
asset_df.reset_index(inplace=True)

# plot color list
color_list = ['Light Red','Blue','Red']

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

for i, term_name in enumerate(term_names):
    legend_label = term_name
    x_label = 'Work Days Since Innauguration'
    y_label = '% Indexed to Inauguration Day'
    x_data = asset_df[asset_df['Term_name'] == term_names[i]]['Date'] - asset_df[asset_df['Term_name'] == term_names[i]]['Date'].iloc[0]
    x_trim = x_data[x_data < f'{days_out} days 00:00:00']
    x_days = x_trim.dt.days.astype(str) + ' days'
    x = [day for day in range(0,len(x_days))]
    y = asset_df[asset_df['Term_name'] == term_names[i]]['GSPC Indexed'].values
    plot_column(ax, x, y[:x[-1]+1], x_label, y_label, color_list[i], legend_label)
plt.show()