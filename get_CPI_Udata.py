# Function to get Consumer Price Index (CPI-U) via the public data API

import requests
import json
import prettytable
import pandas as pd

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

def get_CPI(start_terms, end_terms):
    
    
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
            CPI_df.to_csv(f'{seriesId}_CPI_file.csv', index=False)

        # output = open(seriesId + '.txt','w')
        # output.write (x.get_string())
        # output.close()
        
    

if __name__ == "__main__":
    get_CPI()