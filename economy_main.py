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

# today = dt.today()
# today_str = today.strftime('%Y-%m-%d')
# term_names =['trump1','biden','trump2']
# start_terms = ['2017-01-20',
#                '2021-01-20',
#                '2025-01-20']
# end_terms = ['2021-01-20',
#              '2025-01-20',
#              today_str]
# assets = ['^GSPC']
# x_label = 'Work Days Since Innauguration'
# y_label = '% Indexed to Inauguration Day'

# function to plat data
def plot_column(ax, x, y, xlabel, ylabel, line_color, legend_label):
    # Specify color pallate suitable for people with colorblindness
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
    
    # plot parameters
    use_color = colorblind_palette[line_color]
    ax.plot(x, y, color = use_color, label = legend_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

def economy_track(days_out, assets, term_names, start_terms, end_terms, x_label, y_label):
    
    asset_dict = extract_asset.get_asset(days_out, 
                           assets, 
                           term_names, 
                           start_terms, 
                           end_terms
                           )
    
    for asset in asset_dict.keys():
        asset_df = asset_dict[asset]
    # plot color list
    color_list = ['Light Red','Blue','Red']
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, term_name in enumerate(term_names):
        legend_label = term_name
        # x_label = 'Work Days Since Innauguration'
        # y_label = '% Indexed to Inauguration Day'
        x_data = asset_df[asset_df['Term_name'] == term_names[i]]['Date'] - asset_df[asset_df['Term_name'] == term_names[i]]['Date'].iloc[0]
        x_trim = x_data[x_data < f'{days_out} days 00:00:00']
        x_days = x_trim.dt.days.astype(str) + ' days'
        x = [day for day in range(0,len(x_days))]
        y = asset_df[asset_df['Term_name'] == term_names[i]][f'{asset} Indexed'].values
        plot_column(ax, x, y[:x[-1]+1], x_label, y_label, color_list[i], legend_label)
    plt.show()
    
if __name__ == "__main__":
    # set input parameters
    today = dt.today()
    today_str = today.strftime('%Y-%m-%d')
    term_names =['trump1','biden','trump2']
    start_terms = ['2017-01-20',
                   '2021-01-20',
                   '2025-01-20']
    end_terms = ['2021-01-20',
                 '2025-01-20',
                 today_str]
    assets = ['^GSPC']
    x_label = 'Work Days Since Innauguration'
    y_label = '% Indexed to Inauguration Day'
    days_out = 100
    
    # call main function
    economy_track(days_out, 
                  assets, 
                  term_names, 
                  start_terms, 
                  end_terms, 
                  x_label, 
                  y_label)
    # extract_asset()