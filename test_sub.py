import sys
import asyncio
import os,sys
import shioaji as sj
from shioaji import BidAskFOPv1, Exchange,BidAskSTKv1
import time
import pandas as pd
from datetime import datetime, timedelta,date
# def quote_callback(exchange: Exchange, bidask : BidAskSTKv1):
#         print(f"Exchange : {exchange}, BidAsk: {bidask}")
#def main():
api = sj.Shioaji()
from credentials import api_key, api_secret
#api.quote.set_on_bidask_stk_v1_callback(quote_callback)
from stock_config import Stock_Config
api.login(api_key, api_secret)
start = time.time()
kbars = api.kbars(api.Contracts.Futures[Stock_Config["3037"]][Stock_Config["3037"]+'R1'], start = str(date.today()), end = str(date.today()))
df = pd.DataFrame({**kbars})
df.ts = pd.to_datetime(df.ts)
print(df)
end = time.time()
#print(df[:80]['Close'].to_numpy())
print(len(df))
print(end - start)

# ticks = api.ticks(
#     contract = api.Contracts.Stocks["2330"],
#     date = "2022-12-15"
# )
# print(ticks)
#api.quote.set_on_bidask_fop_v1_callback(quote_callback)
# api.quote.subscribe(
# api.Contracts.Stocks["2330"],
#     quote_type = sj.constant.QuoteType.BidAsk, # or 'bidask'
#     version = sj.constant.QuoteVersion.v1 # or 'v1'
#     )
# try :
#     while True :
#         time.sleep(0.001)
# except KeyboardInterrupt :
#     pass
# except Exception as e :
#     print(e)
    
# if __name__ == '__main__':
#     main()
