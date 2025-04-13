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

def get_asset(days_out, assets, term_names, start_terms, end_terms):
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
    
if __name__ == "__main__":
    get_asset()