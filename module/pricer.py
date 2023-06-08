import asyncio
import decimal
import re
from decimal import Decimal
import logging
import sys
import random
import string
#import telegram
import hashlib
from stock_config import Stock_Config
sys.path.append('./trading-simulator/module')
from prettytable import PrettyTable
import shioaji as sj

# Price precision for submitting orders

def pretty_table(dct):
    table = PrettyTable(['Key', 'Value'])
    for key, val in dct.items():
        table.add_row([key, val])
    return table

def round_price_ref(x,PRECISION_PRICE):
    """
    There's probably a faster way to do this...
    """
    return float(Decimal(x).quantize(PRECISION_PRICE))


def trunc_amount_ref(x,PRECISION_AMOUNT):
    """
    There's probably a faster way to do this...
    """
    with decimal.localcontext() as c:
        #        c.rounding = 'ROUND_DOWN'
        return float(Decimal(x).quantize(PRECISION_AMOUNT))
def round_price_target(x,PRECISION_PRICE):
    """
    There's probably a faster way to do this...
    """
    return float(Decimal(x).quantize(PRECISION_PRICE))


def trunc_amount_target(x,PRECISION_AMOUNT):
    """
    There's probably a faster way to do this...
    """
    with decimal.localcontext() as c:
        #        c.rounding = 'ROUND_DOWN'
        return float(Decimal(x).quantize(PRECISION_AMOUNT))


def side_to_price(side, x):
    neg = x * (-1)
    if side == "BUY":
        return x
    elif side == "SELL":
        return neg



class Pricer:
    active_orders = {}

    def __init__(self, api, ref_symbol, target_symbol, logger,configs):
        self.api = api
        self.ref_symbol = ref_symbol
        self.target_symbol = target_symbol
        self.log = logger
        self.config = configs
        self.lock = asyncio.Lock()
        
    async def create_open_orders(self, spread_prices):
        print("===== create open orders =====")
        async with self.lock:
            contract = self.api.Contracts.Futures[self.ref_symbol][f"{self.ref_symbol}R1"]
            order = self.api.Order(
                price= float(spread_prices.get_price(self.ref_symbol)),
                quantity= int(spread_prices.get_size(self.ref_symbol)),
                action= sj.constant.Action[spread_prices.get_side(self.ref_symbol)],
                price_type=sj.constant.FuturesPriceType["LMT"],
                order_type=sj.constant.OrderType["ROD"],
                octype=sj.constant.FuturesOCType["Auto"],
                account=self.api.futopt_account,
            )
            trade = self.api.place_order(contract, order)
            print(trade)
            contract = self.api.Contracts.Futures[self.target_symbol][f"{self.target_symbol}R1"]
            order = self.api.Order(
                price= float(spread_prices.get_price(self.target_symbol)),
                quantity= int(spread_prices.get_size(self.target_symbol)),
                action= sj.constant.Action[spread_prices.get_side(self.target_symbol)],
                price_type=sj.constant.FuturesPriceType["LMT"],
                order_type=sj.constant.OrderType["ROD"],
                octype=sj.constant.FuturesOCType["Auto"],
                account=self.api.futopt_account,
            )
            trade = self.api.place_order(contract, order)
            print(trade)
               
            

