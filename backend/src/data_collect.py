import subprocess 
import pandas as pd
import datetime as dt
import json
import pandas_datareader as dr
import pandas_datareader.data as web
import yfinance as yf
import urllib.request
from tiingo import TiingoClient
from newspre import get_rapid_news
yf.pdr_override()


# Get the news
def csv_news(stock):
    news_dict, _ = get_rapid_news(stock)
    if news_dict == None:
        return "empty"
    df = pd.DataFrame(news_dict, index=['news polarity'], dtype=None)
    df = df.transpose()
    return df
    
# Get the prices for the news
def get_high_low(stock):
    """
    input: stock ticker

    output: csv file of the stock prices
    """

    # load the stock data, find the date range to search for
    today = dt.date.today()
    since = today - dt.timedelta(days=400)

    # get price updates from yahoo, then format

    prices = dr.get_data_yahoo(stock, since, today)

    prices = prices.drop(['Volume', 'Adj Close'], axis=1)

    return prices


def concat_meta():
    al_data = pd.read_csv("./data/aluminum.csv")
    cu_data = pd.read_csv("./data/copper.csv")
    zi_data = pd.read_csv("./data/zinc.csv")
    oil_data = pd.read_csv("./data/crude-oil.csv")
    sector_data = pd.read_csv("./data/sector.csv")

    al_data.set_index(pd.to_datetime(al_data['Date']), inplace=True)
    cu_data.set_index(pd.to_datetime(cu_data['Date']), inplace=True)
    zi_data.set_index(pd.to_datetime(zi_data['Date']), inplace=True)
    oil_data.set_index(pd.to_datetime(oil_data['Date']), inplace=True)
    sector_data.set_index(pd.to_datetime(sector_data['Date']), inplace=True)
    al_data.drop(columns=['Date', 'Open','High', 'Low', 'Vol.', 'Change %'], inplace=True)
    cu_data.drop(columns=['Date','Open', 'High', 'Low', 'Vol.', 'Change %'],inplace=True)
    zi_data.drop(columns=['Date','Open', 'High', 'Low', 'Vol.', 'Change %'],inplace=True)
    oil_data.drop(columns=['Date','Open', 'High', 'Low', 'Vol.', 'Change %'],inplace=True)
    sector_data.drop(columns=['Date', 'Open','High', 'Low', 'Vol.', 'Change %'], inplace=True)
    merge=pd.merge(cu_data,zi_data, how='inner', left_index=True, right_index=True)
    merge1=pd.merge(merge,al_data, how='inner', left_index=True, right_index=True)
    merge2=pd.merge(merge1,oil_data, how='inner', left_index=True, right_index=True)
    merge3=pd.merge(merge2,sector_data, how='inner', left_index=True, right_index=True)
    #merge3.rename(columns={ merge2.columns[0]: "Cu price", merge2.columns[1]:'Zi data', merge2.columns[2]: 'Al data', merge2.columns[3]: 'oil_data'}, inplace = True)
    return merge3

good_ones = ["VSLR", "RUN", "CSIQ","PEGI","FSLR","SPWR","JKS","ENPH","SEDG", "INE", "TSLA","AQN","DQ","AZRE","ASTI","FP", "YGEHY","SUNW","RGSE","EVSI"]
meta = concat_meta()
for ticker in good_ones:
    try: 
        f = get_high_low(ticker)
        coo = pd.merge(f, meta, how="inner", left_index=True, right_index=True)
        coo['news polarity'] = 0
        nice = csv_news(ticker)

        cols =list(coo.columns.values)
        print(cols)
        for col in range(0, 9):
            for j in range(len(coo.Price) - 1):
                try:
                    coo.iloc[j, col] = float(coo.iloc[j, col].replace(",",""))
                except AttributeError:
                    continue


        coo.to_csv(path_or_buf="./data/prices/{}.csv".format(ticker))
        """if type(nice) == str:
            coo.to_csv(path_or_buf="./data/prices/{}.csv".format(ticker))
        else:
            print(ticker)
            for index, row in nice.iterrows():
                print(row['news polarity'])
                print(index)
                print(row["Name"])
                coo[index]['news polarity'] = row['news polarity']
            coo.to_csv(path_or_buf="./data/prices/{}.csv".format(ticker))"""
    except KeyError or pandas_datareader._utils.RemoteDataError or TypeError:
        continue
#out = subprocess.check_output(["ruby", "stocks.rb", "VSLR"])
