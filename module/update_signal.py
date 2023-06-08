import numpy as np
import collections
import time
import pandas as pd
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
from datetime import timedelta, datetime, date
import os
from module.integer import num_weight
from module.Johenson_class import Johansen
from decimal import Decimal
import decimal
import math
dtype = {
    'S1': str,
    'S2': str,
    'VECMQ': float,
    'mu': float,
    'Johansen_slope': float,
    'stdev': float,
    'model': int,
    'w1': float,
    'w2': float
}
CLOSE_POSITION = {
    "Buy": "Sell",
    "Sell": "Buy"
}


def makehash():
    return collections.defaultdict(makehash)
def find_decimals(value):
    return (abs(decimal.Decimal(str(value)).as_tuple().exponent))
def round_price(x, precision_price):
    return float(Decimal(math.ceil(x * 10 **(find_decimals(precision_price)))/10 **(find_decimals(precision_price))).quantize(precision_price))


def trunc_amount(x, precision_amount):
    return float(Decimal(math.ceil(x * 10 **(find_decimals(precision_amount)))/10 **(find_decimals(precision_amount))).quantize(precision_amount))

class SpreadQuotes:
    spread_price = makehash()
    spread_size = makehash()
    spread_symbol = makehash()
    def __init__(self,ref_symbol,target_symbol):
        self.ref = ref_symbol
        self.target = target_symbol
    def set_size(self, symbol, size):
        assert symbol in [self.ref, self.target]
        self.spread_size[symbol] = size

    def get_size(self, symbol):
        assert symbol in [self.ref, self.target]
        return self.spread_size[symbol]

    def set_price(self, symbol, price):
        self.spread_price[symbol] = price

    def get_price(self, symbol):
        assert symbol in [self.ref, self.target]
        return self.spread_price[symbol]

    def set_side(self, symbol, side):
        self.spread_symbol[symbol] = side

    def get_side(self, symbol):
        assert symbol in [self.ref, self.target]
        return self.spread_symbol[symbol]


class Spreads:
    def __init__(self, window_size):
        self.xs = np.zeros(window_size)
        self.window_size = window_size
        self.index = 0
        self.is_warmed_up = False

    def update(self, x):

        if self.index == self.window_size:
            self.xs = shift(self.xs, -1, cval=0)
            self.index = self.window_size - 1
        self.xs[self.index % self.window_size] = x
        # print(self.xs)
        if self.index == self.window_size - 1:
            self.is_warmed_up = True
        self.index += 1


class Predictor:
    
    five_min_timestamp_1 = 0
    five_min_timestamp_2 = 0
    sec_timestamp_1 = 0
    sec_timestamp_2 = 0

    def __init__(self, window_size, ref_symbol, target_symbol, slippage,log,trading_log,config,api):
        self.window_size = window_size
        self.ref_symbol = ref_symbol
        self.target_symbol = target_symbol
        self.config = config
        self.ref_spreads = Spreads(self.window_size)
        self.target_spreads = Spreads(self.window_size)
        self.ref_timestamp = 0
        self.target_timestamp = 0
        self.slippage = slippage
        self.spread_quotes = SpreadQuotes(self.ref_symbol,self.target_symbol)
        self.logger = log
        self.trading_log = trading_log
        self.position = 0

        self.ref_size = 0
        self.target_size = 0
        self.cointegration_check = False
        self.timestamp_check = False
        self.cointegration_upline = []
        self.cointegration_downline = []
        self.api = api
        self.check = False
    def get_asks(self, orderbook):
        ref_ask = None
        target_ask = None
        if orderbook[self.ref_symbol]['sellQuote'] and orderbook[self.target_symbol]['sellQuote']:
            ref_ask = float(orderbook[self.ref_symbol]
                            ['sellQuote'][0]['price'])
            target_ask = float(
                orderbook[self.target_symbol]['sellQuote'][0]['price'])
        return ref_ask, target_ask

    def get_bids(self, orderbook):
        ref_bid = None
        target_bid = None
        if orderbook[self.ref_symbol]['buyQuote'] and orderbook[self.target_symbol]['buyQuote']:
            ref_bid = float(orderbook[self.ref_symbol]['buyQuote'][0]['price'])
            target_bid = float(
                orderbook[self.target_symbol]['buyQuote'][0]['price'])
        return ref_bid, target_bid
    def get_level_asks(self, orderbook):
        ref_ask = None
        target_ask = None
        if orderbook[self.ref_symbol]['sellQuote'] and orderbook[self.target_symbol]['sellQuote']:
            ref_ask = (float(orderbook[self.ref_symbol]['sellQuote'][0]['price'][0]) + float(orderbook[self.ref_symbol]['sellQuote'][0]['price'][1]) + float(orderbook[self.ref_symbol]['sellQuote'][0]['price'][2])) / 3
            target_ask = (float(orderbook[self.target_symbol]['sellQuote'][0]['price'][0]) + float(orderbook[self.target_symbol]['sellQuote'][0]['price'][1]) + float(orderbook[self.target_symbol]['sellQuote'][0]['price'][2])) / 3
        return ref_ask, target_ask

    def get_level_bids(self, orderbook):
        ref_bid = None
        target_bid = None
        if orderbook[self.ref_symbol]['buyQuote'] and orderbook[self.target_symbol]['buyQuote']:
            ref_bid = (float(orderbook[self.ref_symbol]['buyQuote'][0]['price'][0]) + float(orderbook[self.ref_symbol]['buyQuote'][0]['price'][1]) + float(orderbook[self.ref_symbol]['buyQuote'][0]['price'][2])) / 3

            target_bid = (float(orderbook[self.target_symbol]['buyQuote'][0]['price'][0]) + float(orderbook[self.target_symbol]['buyQuote'][0]['price'][1]) + float(orderbook[self.target_symbol]['buyQuote'][0]['price'][2])) / 3
        return ref_bid, target_bid

    def update_spreads(self, orderbook):
        if self.ref_symbol in orderbook and self.target_symbol in orderbook :
            ref_ask, target_ask = self.get_asks(orderbook)
            ref_bid, target_bid = self.get_bids(orderbook)
            ref_mid_price = (ref_ask + ref_bid) / 2  # ref mid price
            target_mid_price = (target_ask + target_bid) / \
                2  # target mid price
            #self.logger.info(f"REF : {self.ref_symbol} | {ref_mid_price} , TARGET : {self.target_symbol} | {target_mid_price}")
            if ref_ask and target_ask and ref_bid and target_bid:
                self.ref_spreads.update(ref_mid_price)
                self.target_spreads.update(target_mid_price)

    def slippage_number(self, x, size):
        neg = x * (-1)
        if self.position == -1:
            return neg if size > 0 else x
        elif self.position == 1:
            return neg if size < 0 else x

    def side_determination(self, size):
        if self.position == -1:
            return "Sell" if size > 0 else "Buy"
        elif self.position == 1:
            return "Sell" if size < 0 else "Buy"

    def open_Quotes_setting(self, ref_trade_price, target_trade_price,timestamp):
        slippage = self.slippage
        # turn into integer
        self.ref_size ,self.target_size= num_weight(self.table["w1"], self.table["w2"], ref_trade_price,
                                    target_trade_price, self.table["max_hold"], self.table["capital"])
        self.spread_quotes.set_price(
            self.ref_symbol, ref_trade_price)
        self.spread_quotes.set_price(
            self.target_symbol, target_trade_price)
        self.spread_quotes.set_size(
            self.ref_symbol, abs(self.ref_size))
        self.spread_quotes.set_size(
            self.target_symbol, abs(self.target_size))
        self.spread_quotes.set_side(
            self.ref_symbol, self.side_determination(self.ref_size)
        )
        self.spread_quotes.set_side(
            self.target_symbol, self.side_determination(self.target_size)
        )
        self.trading_log.fills('Taiwan', str(timestamp), self.ref_symbol,
                           'LIMIT', self.side_determination(self.ref_size), round_price(ref_trade_price,self.config.PRECISION_PRICE_REF), trunc_amount(abs(self.ref_size),self.config.PRECISION_AMOUNT_REF))
        self.trading_log.fills('Taiwan',  str(timestamp), self.target_symbol,
                           'LIMIT', self.side_determination(self.target_size), round_price(target_trade_price,self.config.PRECISION_PRICE_TARGET), trunc_amount(abs(self.target_size),self.config.PRECISION_AMOUNT_TARGET))

    def close_Quotes_setting(self, ref_trade_price, target_trade_price,timestamp):
        slippage = self.slippage

        # up -> size < 0 -> buy -> ask
        self.spread_quotes.set_price(
            self.ref_symbol, ref_trade_price)
        self.spread_quotes.set_price(
            self.target_symbol, target_trade_price)
        self.spread_quotes.set_size(
            self.ref_symbol, abs(self.ref_size))
        self.spread_quotes.set_size(
            self.target_symbol, abs(self.target_size))
        self.spread_quotes.set_side(
            self.ref_symbol, CLOSE_POSITION[self.side_determination(
                self.ref_size)]
        )
        self.spread_quotes.set_side(
            self.target_symbol, CLOSE_POSITION[self.side_determination(
                self.target_size)]
        )
        self.trading_log.fills('Taiwan',str(timestamp), self.ref_symbol,
                           'LIMIT', CLOSE_POSITION[self.side_determination(self.ref_size)], round_price(ref_trade_price,self.config.PRECISION_PRICE_REF), trunc_amount(abs(self.ref_size),self.config.PRECISION_AMOUNT_REF))
        self.trading_log.fills('Taiwan', str(timestamp), self.target_symbol,
                           'LIMIT', CLOSE_POSITION[self.side_determination(self.target_size)], round_price(target_trade_price,self.config.PRECISION_PRICE_TARGET), trunc_amount(abs(self.target_size),self.config.PRECISION_AMOUNT_TARGET))
        self.position = 888

    def draw_pictrue(self,open_threshold,stop_loss_threshold,stamp,POS,trade_time):
        path_to_image = "./trading_position_pic/"
        path = f'{path_to_image}{self.ref_symbol}_{self.target_symbol}_PIC/' 
        isExist = os.path.exists(path)
        trade_time = trade_time.date()
        if not isExist:    
            # Create a new directory because it does not exist 
            os.makedirs(path)
            print("The new directory is created!")
        curDT = datetime.now()
        time = curDT.strftime("%Y%m%d%H%M")
        sp =  self.table['w1'] * np.log(self.ref_spreads.xs) + self.table['w2'] * np.log(self.target_spreads.xs)
        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax1.plot(sp, color='tab:blue', alpha=0.75)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.hlines(open_threshold * self.table['stdev'] + self.table['mu'], 0, len(sp) + 10,'b')
        #ax1.hlines(stop_loss_threshold * self.table['stdev'] + self.table['mu'], 0, len(sp) + 10, 'b') 
        ax1.hlines(self.table['mu'] - open_threshold * self.table['stdev'], 0, len(sp) + 10,'b')
        #ax1.hlines(self.table['mu'] - stop_loss_threshold * self.table['stdev'], 0, len(sp) + 10, 'b') 
        ax1.hlines(self.table['mu'], 0, len(sp) + 10, 'black') 
        ax1.scatter(len(sp) + 1 ,stamp, color='g', edgecolors='r', marker='o')
        #ax1.text(3,-3,f"w1 = {self.table['w1']}\nw2 = {self.table['w2']}\nstd = {self.table['stdev']}\nmu = {self.table['mu']}")
        #ax1.text(0,0,f"ref : {ref} , bid : {bid}")
        ax1.text(0,self.table['mu'],f"w1 : {self.table['w1']} , w2 : {self.table['w2']}")
        # ax2 = ax1.twinx()
        # ax3 = ax1.twinx()
        ref_return = np.insert(np.diff(self.ref_spreads.xs)/ self.ref_spreads.xs[ : -1],0,0)
        target_return = np.insert(np.diff(self.target_spreads.xs)/ self.target_spreads.xs[ : -1],0,0)
        # ax2.plot(ref_return,color='tab:orange',alpha=0.75) #np.diff(a) / a[:-1]
        # ax3.plot(target_return,color='black', alpha=0.75)
        # ax2.tick_params(axis='y', labelcolor='tab:orange')
        # ax3.tick_params(axis='y', labelcolor='black')
        if POS == 'topopen' or POS == 'downopen':
            plt.savefig(path + str(self.ref_symbol) + '_' + str(self.target_symbol) + '_' +POS+'spread_'+str(trade_time)+'.png')
        elif POS == 'close':
            plt.savefig(path + str(self.ref_symbol) + '_' + str(self.target_symbol) + '_' +POS+'spread_'+ str(trade_time)+'.png')
        self.count += 1
        plt.close()

    def get_target_spread_price(self, orderbook, open_threshold, stop_loss_threshold,cointegration_table):
            ref_ask, target_ask = self.get_asks(orderbook)
            ref_bid, target_bid = self.get_bids(orderbook)
            ref_mid_price = (ref_ask + ref_bid) / 2  # ref mid price
            target_mid_price = (target_ask + target_bid) / 2
            #print("目前部位 : ",self.position)
            t = datetime.now()
            if self.position == 0 and self.cointegration_check is False and self.check is False:
                print("in cointegration check")
                mu, stdev, model, w1, w2 = float(cointegration_table['mu']),float(cointegration_table['stdev']),int(cointegration_table['model']),float(cointegration_table['w1']),float(cointegration_table['w2'])

                if model > 0 and model < 4 and w1 * w2 < 0 :
                    self.cointegration_check = True
                    self.table = {
                        "w1" : w1,
                        "w2" : w2,
                        "mu" : mu,
                        "stdev" : stdev,
                        "model" : model,
                        "max_hold" : 5,
                        "capital" : 1000000,
                    }
                self.check = True
            if self.position == 0 and self.cointegration_check == True:
                
                if self.table["w1"] < 0 and self.table["w2"] > 0:
                    spread_stamp_up = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_bid)
                    spread_stamp_down = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_ask)

                elif self.table["w1"] > 0 and self.table["w2"] < 0:
                    spread_stamp_up = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_ask)
                    spread_stamp_down = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_bid)

                elif self.table["w1"] > 0 and self.table["w2"] > 0:
                    spread_stamp_up = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_bid)
                    spread_stamp_down = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_ask)

                elif self.table["w1"] < 0 and self.table["w2"] < 0:
                    spread_stamp_up = self.table["w1"] * \
                        np.log(ref_ask) + \
                        self.table["w2"] * np.log(target_ask)
                    spread_stamp_down = self.table["w1"] * \
                        np.log(ref_bid) + \
                        self.table["w2"] * np.log(target_bid)
                #print("spread_stamp_up :",spread_stamp_up,"spread_stamp_down :",spread_stamp_down)
                if spread_stamp_up > open_threshold * self.table['stdev'] + self.table['mu'] and spread_stamp_up < self.table["mu"] + self.table["stdev"] * stop_loss_threshold:

                    self.position = -1
                    self.logger.info(
                        f"上開倉")
                    if self.table['w1'] < 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_ask, target_bid,t)
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_up,'topopen',t)
                        return self.spread_quotes
                    elif self.table['w1'] > 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_bid, target_ask,t)
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_up,'topopen',t)

                        return self.spread_quotes
                    elif self.table['w1'] < 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_ask, target_ask,t)

                        print(ref_ask, target_ask)
                        return self.spread_quotes
                    elif self.table['w1'] > 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_bid, target_bid,t)

                        print(ref_bid, target_bid)
                        return self.spread_quotes

                elif spread_stamp_down < self.table['mu'] - open_threshold * self.table['stdev'] and spread_stamp_down > self.table["mu"] - self.table["stdev"] * stop_loss_threshold:
                    self.position = 1
                    #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'open')
                    self.logger.info(
                        f"下開倉")
                    if self.table["w1"] < 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_bid, target_ask,t)
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'downopen',t)
                        #self.draw_pictrue(ref_bid,target_ask,open_threshold,stop_loss_threshold,spread_stamp_down,'open')

                        return self.spread_quotes
                    elif self.table["w1"] > 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_ask, target_bid,t)
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp_down,'downopen',t)
                        #self.draw_pictrue(ref_ask,target_bid,open_threshold,stop_loss_threshold,spread_stamp_down,'open')
                        return self.spread_quotes
                    elif self.table["w1"] < 0 and self.table['w2'] < 0:
                        self.open_Quotes_setting(ref_bid, target_bid,t)

                        print(ref_bid, target_bid)
                        return self.spread_quotes
                    elif self.table["w1"] > 0 and self.table['w2'] > 0:
                        self.open_Quotes_setting(ref_ask, target_ask,t)

                        print(ref_ask, target_ask)
                        return self.spread_quotes
            elif self.position != 0:

                if self.position == -1:
                    if self.ref_size < 0 and self.target_size > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_ask)

                    elif self.ref_size > 0 and self.target_size < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_bid)

                    elif self.ref_size > 0 and self.target_size > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_ask)

                    elif self.ref_size < 0 and self.target_size < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_bid)
                    if spread_stamp < self.table['mu']:
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        self.logger.info("上開倉正常平倉")
                        self.cointegration_check = False
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close',t)
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_ask,t)
                            #self.draw_pictrue(ref_bid,target_ask,open_threshold,stop_loss_threshold,spread_stamp,'close')
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_bid,t)
                            #self.draw_pictrue(ref_ask,target_bid,open_threshold,stop_loss_threshold,spread_stamp,'close')
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_ask,t)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_bid,t)

                            print(ref_bid, target_bid)
                            return self.spread_quotes
                    elif spread_stamp > self.table["mu"] + self.table["stdev"] * stop_loss_threshold:
                        self.cointegration_check = False
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        self.logger.info("上開倉停損平倉")
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close',t)
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_ask,t)
                            #self.draw_pictrue(ref_bid,target_ask,open_threshold,stop_loss_threshold,spread_stamp,'close')
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_bid,t)
                            #self.draw_pictrue(ref_ask,target_bid,open_threshold,stop_loss_threshold,spread_stamp,'close')
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_ask,t)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_bid,t)

                            print(ref_bid, target_bid)
                            return self.spread_quotes
                    elif time.strftime("%H:%M:%S") == '13:24:59':
                        self.cointegration_check = False
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        self.logger.info(
                            f"上開倉強制平倉")
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close',t)
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_ask,t)
                            #self.draw_pictrue(ref_bid,target_ask,open_threshold,stop_loss_threshold,spread_stamp,'close')
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_bid,t)
                            #self.draw_pictrue(ref_ask,target_bid,open_threshold,stop_loss_threshold,spread_stamp,'close')
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_ask,t)

                            print(ref_ask, target_ask)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_bid,t)

                            print(ref_bid, target_bid)
                            return self.spread_quote
                elif self.position == 1:
                    if self.ref_size < 0 and self.target_size > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_bid)

                    elif self.ref_size > 0 and self.target_size < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_ask)

                    elif self.ref_size > 0 and self.target_size > 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_bid) + \
                            self.table["w2"] * np.log(target_bid)

                    elif self.ref_size < 0 and self.target_size < 0:
                        spread_stamp = self.table["w1"] * \
                            np.log(ref_ask) + \
                            self.table["w2"] * np.log(target_ask)
                    if spread_stamp > self.table['mu']:
                        self.cointegration_check = False
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close',t)
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        self.logger.info(
                            f"下開倉正常平倉 : Ref Size : {self.ref_size} Ref Price :{ref_mid_price} Target Size : {self.target_size} Target Price :{target_mid_price}")
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_bid,t)
                            #self.draw_pictrue(ref_ask,target_bid,open_threshold,stop_loss_threshold,spread_stamp,'close')
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_ask,t)
                            #self.draw_pictrue(ref_bid,target_ask,open_threshold,stop_loss_threshold,spread_stamp,'close')
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_bid,t)
                            print(ref_bid, target_bid)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_ask,t)
                            print(ref_ask, target_ask)
                            return self.spread_quotes
                    elif spread_stamp < self.table["mu"] - self.table["stdev"] * stop_loss_threshold:
                        self.cointegration_check = False
                        #self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close')
                        self.logger.info(
                            f"下開倉停損平倉")
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close',t)
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_bid,t)
                           #self.draw_pictrue(ref_ask,target_bid,open_threshold,stop_loss_threshold,spread_stamp,'close')
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_ask,t)
                            #self.draw_pictrue(ref_bid,target_ask,open_threshold,stop_loss_threshold,spread_stamp,'close')
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_bid,t)
                            print(ref_bid, target_bid)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_ask,t)
                            print(ref_ask, target_ask)
                            return self.spread_quotes
                    elif time.strftime("%H:%M:%S") == '13:24:50':
                        self.logger.info(
                            f"下開倉強制平倉")
                        self.draw_pictrue(open_threshold,stop_loss_threshold,spread_stamp,'close',t)
                        if self.ref_size < 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_ask, target_bid,t)
                            #self.draw_pictrue(ref_ask,target_bid,open_threshold,stop_loss_threshold,spread_stamp,'close')
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_bid, target_ask,t)
                            #self.draw_pictrue(ref_bid,target_ask,open_threshold,stop_loss_threshold,spread_stamp,'close')
                            return self.spread_quotes
                        elif self.ref_size > 0 and self.target_size > 0:
                            self.close_Quotes_setting(ref_bid, target_bid,t)
                            print(ref_bid, target_bid)
                            return self.spread_quotes
                        elif self.ref_size < 0 and self.target_size < 0:
                            self.close_Quotes_setting(ref_ask, target_ask,t)
                            print(ref_ask, target_ask)
                            return self.spread_quotes