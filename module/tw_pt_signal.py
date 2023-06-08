import asyncio
from module.pricer import Pricer
from module.update_signal import Predictor
import traceback
import shioaji as sj
from shioaji import BidAskFOPv1, Exchange,BidAskSTKv1,TickSTKv1
from datetime import timedelta,datetime
import logger
import json
import os
from module.log_format import SaveLog

def datetime_range(start, end, delta):
    current = start
    while current < end :
        yield current
        current += delta


class TF_pair_trading:
    # chat_id = ''
    # bot = telegram.Bot(token=(''))

    orderbook = {}
    orderbook_1min = {}
    trades = {}
    Ref_SeqNum = 0
    Target_SeqNum = 0
    signal_path = '/home/allenkuo/tw_pt_stock_future/signal_dict/'
    def __init__(self, api, order_api, config,s_to_f_config,path):
        self.api = api
        self.order_api = order_api
        self.config = config
        self.s_to_f_config = s_to_f_config
        self.f_to_s_config = {v : k for k,v in self.s_to_f_config.items()}
        self.log = logger.get_logger(f"TW_Pair_Trading_{config.REFERENCE_SYMBOL}_{config.TARGET_SYMBOL}_logger")
        self.trading_log = SaveLog(
            "Allen",
            "PairTrading",
            f"{self.config.REFERENCE_SYMBOL}{self.config.TARGET_SYMBOL}_{int(self.config.TEST_SECOND/60)}min_{self.config.OPEN_THRESHOLD}_{self.config.STOP_LOSS_THRESHOLD}",
            path
        )
        self.predictor = Predictor(
            window_size=self.config.MA_WINDOW_SIZE,
            ref_symbol=config.REFERENCE_SYMBOL,
            target_symbol=config.TARGET_SYMBOL,
            slippage=config.SLIPPAGE,
            log = self.log,
            trading_log=self.trading_log,
            config=self.config,
            api = self.api
        )
        self.pricer = Pricer(
            order_api,
            config.REFERENCE_SYMBOL,
            config.TARGET_SYMBOL,
            self.log,
            self.config
        )
        self.spread_prices = None
        self.remember_quotos = None
        self.readable = False
        self.signal_file = f"TW_{self.config.REFERENCE_SYMBOL}_{self.config.TARGET_SYMBOL}_signal.json"
        self.cointegration_table = None
    def order_cb(self,stat, msg):
        print('my_order_callback')
        self.log.info(stat, msg)
    def quote_callback(self,exchange:Exchange, bidask:BidAskFOPv1):
        #print(f"Exchange: {exchange}, BidAsk: {bidask}")
        symbol = bidask.code
        symbol = f'{symbol[:-2]}'
        if symbol == self.s_to_f_config[self.config.REFERENCE_SYMBOL]:
            self.orderbook[self.config.REFERENCE_SYMBOL] = {
                                'buyQuote': [{'price': float(bidask.bid_price[0]) , 'size': bidask.bid_volume[0]}],
                                'sellQuote': [{'price': float(bidask.ask_price[0]), 'size':bidask.ask_volume[0]}],}  
        elif symbol == self.s_to_f_config[self.config.TARGET_SYMBOL]:
            self.orderbook[self.config.TARGET_SYMBOL] = {
                                'buyQuote': [{'price': float(bidask.bid_price[0]) , 'size': bidask.bid_volume[0]}],
                                'sellQuote': [{'price': float(bidask.ask_price[0]) , 'size':bidask.ask_volume[0]}],}
        print(self.orderbook)
        self.predictor.update_spreads(self.orderbook)
    def read_cointegration_table(self):
        if os.path.exists(self.signal_path + self.signal_file):
    # 检查文件大小是否为 0
            if os.path.getsize(self.signal_path + self.signal_file) > 0:
                # 读取 JSON 数据
                with open(self.signal_path + self.signal_file, 'r') as file:
                    json_data = file.read()

                data_dict = json.loads(json_data)
                return data_dict
        return None
    async def check_trading(self):
        while True :
            #self.log.info("check trading")
            if not self.readable :
                self.cointegration_table = self.read_cointegration_table()
                print(self.cointegration_table)
                if self.cointegration_table is not None:
                    self.readable = True
            if self.readable and self.config.REFERENCE_SYMBOL in self.orderbook and self.config.TARGET_SYMBOL in self.orderbook and float(self.cointegration_table['w1']) != 0 and float(self.cointegration_table['w2']) != 0:
                #print("check spread price ----------------------")
                self.spread_prices = self.predictor.get_target_spread_price(
                                        orderbook=self.orderbook,
                                        open_threshold=self.config.OPEN_THRESHOLD,
                                        stop_loss_threshold=self.config.STOP_LOSS_THRESHOLD,
                                        cointegration_table = self.cointegration_table,
                                    )
                if self.spread_prices is not None:
                    print("create order")
                    self.pricer.create_open_orders(self.spread_prices)
                
            await asyncio.sleep(0.001)  
    
    def Update_orderbook(self,symbol):
        # self.api.quote.subscribe(
        # self.api.Contracts.Stocks[symbol],
        # quote_type = sj.constant.QuoteType.BidAsk, # or 'bidask'
        # version = sj.constant.QuoteVersion.v1 # or 'v1'
        # )
        self.api.quote.subscribe(
        self.api.Contracts.Futures[self.s_to_f_config[symbol]][f"{self.s_to_f_config[symbol]}R1"],
        quote_type = sj.constant.QuoteType.BidAsk, # or 'bidask'
        version = sj.constant.QuoteVersion.v1 # or 'v1'
        )
        self.api.quote.set_on_bidask_fop_v1_callback(self.quote_callback)

    async def execute(self):
        while True:
            try:
                self.api.set_order_callback(self.order_cb)
                #self.api.quote.set_on_bidask_fop_v1_callback(self.quote_callback)
                self.Update_orderbook(self.config.REFERENCE_SYMBOL)
                self.Update_orderbook(self.config.TARGET_SYMBOL)
                trading_signal = asyncio.create_task(self.check_trading())
                print("start trading")
                await asyncio.gather(
                    trading_signal,
                )

            
            except Exception as e:
                print(traceback.format_exc())
                self.log.info(e)
                trading_signal.cancel()
                #continue