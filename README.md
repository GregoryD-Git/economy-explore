![Markdown Logo](https://markdown-here.com/img/icon256.png)

# economy-explore
A practice in obtaining data from the web and creating informative visuals to better understand how presidencies impact the economy

> **KEY MODULES AND PROCESSES USED**
'''
> - yfinance
> - pandas
> - nympy
> - datetime
> - matplotlib.pyplot
'''

**KEY METHODS USED**
> - Data extraction and manipulation
> - Data visualization
> - Data communication

There are (two) reports demonstrating key economic factors that demonstrate the "health of the economy", over the course of the repspective presidential terms. 

First, is the S&P500 
## S&P500 data presentation
- Looking at the closing price of the S&P500 over the course of a presidency 
can help understand how the market is affected by presidential policy
- The S&P500 is considered an index market for the general state of the 
market as a whole
- Tracking the markets closing costs over time, indexed to inauguration day
can help to understand how the market changes over time

To use the S&P500, the python modul yfinance is used
'import yfinance as yf'

---
***analysis ideas***
- Determine how presidential political affiliation impacts overall market
progress: 
### NULL HYPOTHESES
    *H0-political affiliation does not predict overall change in market price
    from beginning to end of a presidential term
    *H1-politial affilitation does not predict the overall rate of change in 
    market price over the course of a presidency


