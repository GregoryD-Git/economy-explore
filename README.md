<h1 style="display: flex; align-items: center;">
  <img src=https://markdown-here.com/img/icon256.png alt="Description" width="50" height="50" style="margin-left: 10px;">
  Economy-Explore
</h1>

A practice in obtaining data from the web and creating informative visuals to better understand how presidencies impact the economy

**KEY METHODS, MODULES, AND PROCESSES USED**
> 1. Data extraction via python module and web API
> 2. yfinance, pandas, numpy, matplotlib
> 3. Data extraction updated day-of

**DESCRIPTION:** Here, (two) reports are presented, demonstrating key economic factors that demonstrate the "health of the economy", over the course of the respective presidential terms. 

## S&P500 data presentation
- Looking at the closing price of the S&P500 over the course of a presidency 
can help understand how the market is affected by presidential policy
- The S&P500 is considered an index market for the general state of the 
market as a whole
- Tracking the markets closing costs over time, indexed to inauguration day
can help to understand how the market changes during the course of a presidency

To use the S&P500, the python modul yfinance is used
<pre>
  python import yfinance as yf
</pre>

### S&P500 Indexed to First Day in Office
![Sample Plot](economy-explore_asset.png)

### Short summary
This graph illustrates the performance of the S&P 500, normalized to each president's first day in office, across three different presidencies. It highlights trends in the market during their respective terms, providing a comparative perspective on economic patterns during each administration. 
---
***analysis ideas***
- Determine how presidential political affiliation impacts overall market
progress: 
### NULL HYPOTHESES
    *H0-political affiliation does not predict overall change in market price
    from beginning to end of a presidential term
    *H1-politial affilitation does not predict the overall rate of change in 
    market price over the course of a presidency

## Consumer Price Index
Gathered from the API of the US Beaureu of Labor Statistics

[Visit the BLS Developers Page for access to the API](https://www.bls.gov/developers/home.htm)


