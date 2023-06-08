import sys
import asyncio
import os,sys
from module.tw_pt import TF_pair_trading
import shioaji as sj
from config import Pair_Trading_Config
from stock_config import Stock_Config
import time
def order_cb(stat, msg):
        print('my_order_callback')
        print(stat, msg)
async def place_order(api,config):
    contract = api.Contracts.Futures[Stock_Config[config.TEST_SYMBOL]][Stock_Config[config.TEST_SYMBOL]+config.FUTURE_DATE_SYMBOL]
    order = api.Order(
                price= float(102.0),
                quantity= int(1),
                action= sj.constant.Action['Buy'],
                price_type=sj.constant.FuturesPriceType["LMT"],
                order_type=sj.constant.OrderType["ROD"],
                octype=sj.constant.FuturesOCType["Auto"],
                account=api.futopt_account,
            )
    trade = api.place_order(contract, order)
    print(trade)
async def place_order2(api,config):
    contract = api.Contracts.Futures[Stock_Config[config.TEST_SYMBOL]][Stock_Config[config.TEST_SYMBOL]+config.FUTURE_DATE_SYMBOL]
    order = api.Order(
                price= float(104.0),
                quantity= int(1),
                action= sj.constant.Action['Buy'],
                price_type=sj.constant.FuturesPriceType["LMT"],
                order_type=sj.constant.OrderType["ROD"],
                octype=sj.constant.FuturesOCType["Auto"],
                account=api.futopt_account,
            )
    trade = api.place_order(contract, order)
    print(trade)
async def main():
    api = sj.Shioaji(simulation = True)
    from credentials import api_key, api_secret
    api.login(api_key, api_secret)
    config = Pair_Trading_Config()
    print(Stock_Config[config.TEST_SYMBOL]+config.FUTURE_DATE_SYMBOL)
    api.set_order_callback(order_cb)
    await place_order(api,config)
    time.sleep(10)
    await place_order2(api,config)

         
    
if __name__ == '__main__':
    asyncio.run(main())