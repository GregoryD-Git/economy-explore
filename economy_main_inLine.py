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
import get_CPI_Udata 

# stock market financial data
import yfinance as yf

# consumer price index data
import requests
import json
import prettytable

##################################### Plot Function ###########################
# function to plat data
def plot_column(ax, x, y, my_title, xlabel, ylabel, line_color, legend_label, width):
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

################################### Stock Market Data #########################

def get_asset(days_out, assets, term_names, start_terms, end_terms):
    # Stock data presentation
    #     - Looking at the closing price of the S&P500 over the course of a presidency 
    #     can help understand how the market is affected by presidential policy
    #     - The S&P500 is considered an index market for the general state of the 
    #     market as a whole
    #     - Tracking the markets closing costs over time, indexed to inauguration day
    #     can help to understand how the market changes over time
        
    #     *analysis ideas???
    #     - Determine how presidential political affiliation impacts overall market
    #     progress: 
    #         NULL HYPOTHESES
    #         *H0-political affiliation does not predict overall change in market price
    #         from beginning to end of a presidential term
    #         *H1-politial affilitation does not predict the overall rate of change in 
    #         market price over the course of a presidency
    
    # Specify how many days out from inauguration we want to see stock data
    # days_out = 100 # number of work days in a year
    
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

############################## Consumer Price Index Data ######################
def get_CPI(start_terms, end_terms):
    # CUUR0000SA0: This refers to the CPI-U (Consumer Price Index for All Urban 
    # Consumers) for all items in the U.S. city average. The "U" indicates that the 
    # data is not seasonally adjusted.
    
    # For seasonally adjusted CPI data from the BLS, you should use series tags 
    # where the third character is "S" (indicating seasonally adjusted data). 
    # For example: CUUS0000SA0: Seasonally adjusted CPI-U 
    # (Consumer Price Index for All Urban Consumers) for all items in the U.S. 
    # city average.
    
    # SUUR0000SA0: This represents the Chained CPI-U (Consumer Price Index for All 
    # Urban Consumers) for all items in the U.S. city average. The "U" also indicates 
    # that the data is not seasonally adjusted.
    
    # month list
    months = ['January','February','March',
             'April','May','June','July',
             'August','September','October',
             'November','December']
    
    for i, start_date in enumerate(start_terms):
        start_year = start_date.split('-')[0]
        # print(f'Start Year: {start_year}')
        end_year = end_terms[i].split('-')[0]
        # print(f'End Year: {end_year}')
        
        headers = {'Content-type': 'application/json'}
        data = json.dumps({"seriesid": ['CUSR0000SA0'],"startyear":start_year, "endyear":end_year})
        p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
        json_data = json.loads(p.text)
        rows = []
        CPI_df = pd.DataFrame()
        
        for series in json_data['Results']['series']:
            x=prettytable.PrettyTable(["series id","year","period","value","footnotes"])
            seriesId = series['seriesID']
            
            for item in series['data']:
                year = item['year']
                period = item['period'] # month
                month = months[int(period[1:]) - 1]
                value = item['value']
                footnotes=""
                for footnote in item['footnotes']:
                    if footnote:
                        footnotes = footnotes + footnote['text'] + ','
                if 'M01' <= period <= 'M12':
                    x.add_row([seriesId,year,month,value,footnotes[0:-1]])
                    rows.append({'year': year, 'month': month, 'value': value})
            
                df = pd.DataFrame(rows)
            CPI_df = pd.concat([CPI_df, df], ignore_index=True, axis=0)
            
            # Export to a .csv file
            # CPI_df.to_csv(f'{seriesId}_CPI_file.csv', index=False)
        return CPI_df
        # output = open(seriesId + '.txt','w')
        # output.write (x.get_string())
        # output.close()
        
def economy_track(days_out, assets, term_names, start_terms, end_terms, x_label, y_label):
    ###########################################################################
    # Extract data for the S&P500
    asset_dict = extract_asset.get_asset(days_out, 
                           assets, 
                           term_names, 
                           start_terms, 
                           end_terms
                           )
    
    for asset in asset_dict.keys():
        asset_df = asset_dict[asset]
    # plot color list
    color_list = ["light_green","dark_green",
                "light_blue",
                "light_red","dark_red"
                ]
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('tableau-colorblind10')
    title = 'S&P 500 Index to First Day in Office'
    
    for i, term_name in enumerate(term_names):
        legend_label = term_name
        # x_label = 'Work Days Since Innauguration'
        # y_label = '% Indexed to Inauguration Day'
        x_data = asset_df[asset_df['Term_name'] == term_names[i]]['Date'] - asset_df[asset_df['Term_name'] == term_names[i]]['Date'].iloc[0]
        x_trim = x_data[x_data < f'{days_out} days 00:00:00']
        x_days = x_trim.dt.days.astype(str) + ' days'
        x = [day for day in range(0,len(x_days))]
        y = asset_df[asset_df['Term_name'] == term_names[i]][f'{asset} Indexed'].values
        width = 1
        if i == len(term_names)-1:
            width = 3
        plot_column(ax, x, y[:x[-1]+1], 
                    title, x_label, y_label, 
                    color_list[i], legend_label, width
                    )
    
    plt.savefig("economy_explore_asset.png")  # Saves the figure to a .png file
    plt.show()
    
    ###########################################################################
    # Extract data for CPI-U
    # using separate function
    # get_CPI_Udata.get_CPI(start_terms, end_terms)
    
    # using function in this script
    CPI_df = get_CPI(start_terms, end_terms)
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('tableau-colorblind10')
    title = 'Consumer Price Index Scaled to First Day in Office'
    
    for i, term_name in enumerate(term_names):
        legend_label = term_name
        # x_label = 'Work Days Since Innauguration'
        # y_label = '% Indexed to Inauguration Day'
        x_data = asset_df[asset_df['Term_name'] == term_names[i]]['Date'] - asset_df[asset_df['Term_name'] == term_names[i]]['Date'].iloc[0]
        x_trim = x_data[x_data < f'{days_out} days 00:00:00']
        x_days = x_trim.dt.days.astype(str) + ' days'
        x = [day for day in range(0,len(x_days))]
        y = asset_df[asset_df['Term_name'] == term_names[i]][f'{asset} Indexed'].values
        width = 1
        if i == len(term_names)-1:
            width = 3
        plot_column(ax, x, y[:x[-1]+1], 
                    title, x_label, y_label, 
                    color_list[i], legend_label, width
                    )
    
    plt.savefig("economy_explore_asset.png")  # Saves the figure to a .png file
    plt.show()
    
if __name__ == "__main__":
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
    days_out = 400
    
    # call main function
    economy_track(days_out, assets, term_names, 
                  start_terms, end_terms, 
                  x_label, y_label)
    # extract_asset()