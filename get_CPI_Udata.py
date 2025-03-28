# Function to get Consumer Price Index (CPI-U) via the public data API

import requests
import json
import prettytable

def get_CPI(start_year, end_year):
    headers = {'Content-type': 'application/json'}
    data = json.dumps({"seriesid": ['CUUR0000SA0','SUUR0000SA0'],"startyear":start_year, "endyear":end_year})
    p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
    json_data = json.loads(p.text)
    for series in json_data['Results']['series']:
        x=prettytable.PrettyTable(["series id","year","period","value","footnotes"])
        seriesId = series['seriesID']
        for item in series['data']:
            year = item['year']
            period = item['period']
            value = item['value']
            footnotes=""
            for footnote in item['footnotes']:
                if footnote:
                    footnotes = footnotes + footnote['text'] + ','
            if 'M01' <= period <= 'M12':
                x.add_row([seriesId,year,period,value,footnotes[0:-1]])
        output = open(seriesId + '.txt','w')
        output.write (x.get_string())
        output.close()
        
    

if __name__ == "__main__":
    get_CPI()