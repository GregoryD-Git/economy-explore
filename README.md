<h1 style="display: flex; align-items: center;">
  <img src="mock_stock_market_arrow.png" alt="Stock Market Ticker" width="50" height="50" style="margin-left: 10px;">
  Economy-Explore S&P500
</h1>

**DESCRIPTION:**
This project is meant to demonstrate various data science skills with specific and relatable data visualization for the average person.

The first project is a practice in extracting stock data from a python module and plotting the data to see how the market is impacted over the course of several recent presidencies. 

**KEY METHODS, MODULES, AND PROCESSES USED**
> 1. METHOD: Data extraction via python module into dataframe, plotted using matplotlib
> 2. MODULES: yfinance, pandas, numpy, matplotlib
> 3. PROCESS: Code is executed each day to demonstrate the most recent administrations impact on the stock market relative to prior administrations

## The Economy via the S&P500
The S&P500 is the asset of choice for the following reasons
- Diverse Representation: The index includes 500 of the largest publicly traded companies across various sectors, like tech, healthcare, finance, and energy. This diversity provides insight into how different industries are performing.
- Market Capitalization: Since it's weighted by market cap, the S&P 500 reflects the influence of larger companies that significantly impact the economy. When these giants thrive or struggle, it can signal broader economic trends.
- Global Reach: Many companies in the S&P 500 operate internationally, so the index isn't just a measure of the U.S. economy—it also reflects global economic activity.
- Investor Sentiment: Movements in the S&P 500 often mirror investor confidence. For example, strong performance can suggest optimism about economic growth, while declines might point to concerns or challenges.
- Economic Indicators: The index is frequently used alongside other metrics, like GDP growth, employment rates, and inflation, to provide a fuller picture of economic health.

## How the S&P500 is used here
- Looking at the closing price of the S&P500 over the course of a presidency can help understand how the market is affected by presidential policy
- The S&P500 is considered an index market for the general state of the market as a whole
- Tracking the markets closing costs over time, indexed to inauguration day can help to understand how the market changes during the course of a presidency

## The code
To use the S&P500, the python modul yfinance is used and imported via a custom written function
<pre>
  import matplotlib.pyplot as plt
  from datetime import datetime as dt
  import extract_asset
</pre>
A plot function is set to create a figure using matplotlib
<pre>
  def plot_column(ax, x, y, my_title, xlabel, ylabel, line_color, legend_label, width):
    # Specify color pallate suitable for people with colorblindness
    colorblind_palette = {...
</pre>
Input parameters are set for the terms of interest and an asset is chosen
<pre>
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
</pre>
The function to extract the asset data is called
<pre>
  # Extract data for the S&P500
  asset_dict = extract_asset.get_asset(days_out, 
                         assets, 
                         term_names, 
                         start_terms, 
                         end_terms
                         )
</pre>
A figure object is created, the terms of interest are called and looped through to add data to the figure and plotted
<pre>
  # Create a figure and axes
  fig, ax = plt.subplots(figsize=(10, 6))
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
      plot_column(ax, x, y[:x[-1]+1], 
                  title, x_label, y_label, 
                  color_list[i], legend_label, width
                  )
</pre>
The figure is saved to png format picture for later use
<pre>
  plt.savefig("economy_getStockAsset.png")  # Saves the figure to a .png file
  plt.show()
</pre>

### S&P500 Indexed to First Day in Office
![Sample Plot](economy_S&P500byTerm.png)

## Short summary
- This graph illustrates the performance of the S&P 500, normalized to each president's first day in office, across the chosen presidential terms.
- It highlights trends in the market during those periods, providing a comparative perspective on economic patterns during each administration. 
---
***further analyses to come***
- Look at overall change from beginning to end of term
- Look at market volatility for each term (std)
- Maximum drawdown - largest peak to trough drop in the index to understand risk of loss
- Sharpe ratio - risk-adjusted performance comparing returns to volatility - evaluating whether risk is justified

## Consumer Price Index
Gathered from the API of the US Beaureu of Labor Statistics

[Visit the BLS Developers Page for access to the API](https://www.bls.gov/developers/home.htm)


