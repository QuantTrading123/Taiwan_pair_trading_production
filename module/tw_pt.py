import asyncio
from module.pricer import Pricer
from functools import partial
from module.predictor_change_spreadstamp import Predictor
import traceback
import shioaji as sj
from shioaji import BidAskFOPv1, Exchange,BidAskSTKv1,TickSTKv1
from datetime import timedelta,datetime
import logger
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
    def __init__(self, api, order_api, config,s_to_f_config):
        self.api = api
        self.order_api = order_api
        self.config = config
        self.s_to_f_config = s_to_f_config
        self.f_to_s_config = {v : k for k,v in self.s_to_f_config.items()}
        self.log = logger.get_logger(f"TW_Pair_Trading_{config.REFERENCE_SYMBOL}_{config.TARGET_SYMBOL}_logger")
        self.predictor = Predictor(
            window_size=self.config.MA_WINDOW_SIZE,
            ref_symbol=config.REFERENCE_SYMBOL,
            target_symbol=config.TARGET_SYMBOL,
            slippage=config.SLIPPAGE,
            log = self.log,
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
    def order_cb(self,stat, msg):
        print('my_order_callback')
        self.log.info(stat, msg)
    def quote_callback(self,exchange:Exchange, bidask:BidAskFOPv1):
        #print(f"Exchange: {exchange}, BidAsk: {bidask}")
        symbol = bidask.code
        timestamp = datetime.now()#bidask.datetime#.replace(microsecond = 0)
        symbol = self.f_to_s_config[f'{symbol[:-2]}']
        #print(timestamp)
        #if symbol not in self.orderbook_1min or timestamp - self.orderbook_1min[symbol]['timestamp'] >= timedelta(seconds=self.config.TEST_SECOND):
        if symbol not in self.orderbook_1min or timestamp.second == 0:
            print("in")
            self.orderbook_1min[symbol] = {
                    'buyQuote': [{'price': bidask.bid_price[0], 'size': bidask.bid_volume[0]}],
                            'sellQuote': [{'price': bidask.ask_price[0], 'size':bidask.ask_volume[0]}],
                            'timestamp': timestamp}   
            self.predictor.update_spreads(self.orderbook_1min)
        
        self.orderbook[symbol] = {
                            'buyQuote': [{'price': bidask.bid_price[0], 'size': bidask.bid_volume[0]}],
                            'sellQuote': [{'price': bidask.ask_price[0], 'size':bidask.ask_volume[0]}],
                             'timestamp': timestamp}  
        #print(self.orderbook)
    async def check_trading(self,task_queue):
        while True :
            #self.log.info("check trading")
            #print("in")
            self.spread_prices = self.predictor.get_target_spread_price(
                                    orderbook=self.orderbook,
                                    open_threshold=self.config.OPEN_THRESHOLD,
                                    stop_loss_threshold=self.config.STOP_LOSS_THRESHOLD,
                                )
            #self.log.info("spread quotes : {}".format(self.spread_prices))
            if self.spread_prices \
                and self.predictor.ref_spreads.is_warmed_up \
                and self.predictor.target_spreads.is_warmed_up:
                print("Time to create open orders")
                #await self.pricer.create_open_orders(self.spread_prices)
                await task_queue.put(
                        partial(self.pricer.create_open_orders,
                        self.predictor.spread_quotes)
                    )
            await asyncio.sleep(0.001)  
    
    def Update_orderbook(self,symbol):
        # self.api.quote.subscribe(
        # self.api.Contracts.Stocks[symbol],
        # quote_type = sj.constant.QuoteType.BidAsk, # or 'bidask'
        # version = sj.constant.QuoteVersion.v1 # or 'v1'
        # )
        self.api.quote.subscribe(
        self.api.Contracts.Futures[self.s_to_f_config[symbol]][self.s_to_f_config[symbol]+self.config.FUTURE_DATE_SYMBOL],
        quote_type = sj.constant.QuoteType.BidAsk, # or 'bidask'
        version = sj.constant.QuoteVersion.v1 # or 'v1'
        )
        #self.api.quote.set_on_bidask_fop_v1_callback(self.quote_callback)


    async def execute_task(self, task_queue):
        while True:
            try:
                task = await task_queue.get()
                if asyncio.iscoroutinefunction(task.func):
                    await task()
                else:
                    task()
                task_queue.task_done()
            except Exception as e:
                print(traceback.format_exc())
    async def execute(self):
        while True:
            try:
                task_queue  = asyncio.Queue()
                trade_queue = asyncio.Queue()
                # update_ob_ref = asyncio.create_task(self.Update_orderbook(self.config.REFERENCE_SYMBOL))
                # update_ob_target = asyncio.create_task(self.Update_orderbook(self.config.TARGET_SYMBOL))
                self.api.set_order_callback(self.order_cb)
                self.api.quote.set_on_bidask_fop_v1_callback(self.quote_callback)
                self.Update_orderbook(self.config.REFERENCE_SYMBOL)
                self.Update_orderbook(self.config.TARGET_SYMBOL)
                trading_signal = asyncio.create_task(self.check_trading(task_queue))
                #trading_signal.add_done_callback(self.quote_callback)
                #update_trade = asyncio.create_task(self.Update_Trade(trade_queue))
                tasks = []
                for i in range(2):
                    task = asyncio.create_task(self.execute_task(task_queue))
                    tasks.append(task)
                await asyncio.gather(
                    trading_signal,
                    *tasks
                )

            
            except Exception as e:
                print(traceback.format_exc())
                self.log.info(e)
                trading_signal.cancel()
                #pdate_ob_target.cancel()
                #update_trade.cancel()
                for t in tasks:
                    t.cancel()
                #continue