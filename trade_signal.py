import asyncio
import shioaji as sj
from config import Pair_Trading_Config
from stock_config import Stock_Config
from credentials import api_key, api_secret
import pandas as pd
from datetime import date
from module.Johenson_class import Johansen
import numpy as np
import json
import os
import sys
class TF_pair_trading_signal :
    signal_path = '/home/allenkuo/tw_pt_stock_future/signal_dict/'
    def __init__(self,api,configs) -> None:
        self.api = api
        self.config = configs
        self.signal_file = f"TW_{self.config.REFERENCE_SYMBOL}_{self.config.TARGET_SYMBOL}_signal.json"
        self.window_size = self.config.MA_WINDOW_SIZE
        self.cointegration_test_check = True
        self.trading_date = str(date.today())
    def cointegration_test(self):
        # Get the first defined length data points for reference and target symbols
        kbars_ref = self.api.kbars(self.api.Contracts.Stocks[self.config.REFERENCE_SYMBOL], start = self.trading_date , end = self.trading_date )
        df = pd.DataFrame({**kbars_ref})
        df.ts = pd.to_datetime(df.ts)
        df.set_index('ts', inplace=True)
        df = df[df.index >= f'{self.trading_date} 09:00:00']
        print(df)
        print(len(df))
        if len(df) >= int(self.window_size/60)+16 and self.cointegration_test_check :
            ref_df = df[16:int(self.window_size/60)+16]['Close'].astype(float).to_numpy()
            kbars_target = self.api.kbars(self.api.Contracts.Stocks[self.config.TARGET_SYMBOL], start = self.trading_date , end = self.trading_date )
            df = pd.DataFrame({**kbars_target})
            df.ts = pd.to_datetime(df.ts)
            df.set_index('ts', inplace=True)
            df = df[df.index >= f'{self.trading_date} 09:00:00']
            target_df = df[16:int(self.window_size/60)+16]['Close'].astype(float).to_numpy()
            # Combine the reference and target data into a single data set
            price_series = [[r, t] for r, t in zip(
                ref_df, target_df)]
            price_series = np.array(price_series)
            print(price_series)
            
            # Perform Johanson cointegration test
            jc = Johansen(price_series)
            Johanson_cointegration = jc.execute()
            print(Johanson_cointegration)
            saved_dict = {"mu": Johanson_cointegration[0],"stdev": Johanson_cointegration[1],"model": Johanson_cointegration[2]
                        ,"w1": Johanson_cointegration[3],"w2": Johanson_cointegration[4]}
            json_data = json.dumps(saved_dict)
            self.cointegration_test_check = False

            # 将 JSON 字符串写入文件
            with open(self.signal_path + self.signal_file, 'w') as file:
                file.write(json_data)
            return 1
        return 0

def main(ref_symbol,target_symbol):
    api = sj.Shioaji()
    
    api.login(api_key, api_secret)
    configs = Pair_Trading_Config(ref_symbol,target_symbol)
    tw_pair_trading = TF_pair_trading_signal(api, configs)
    while True:
        check = tw_pair_trading.cointegration_test()
        if check == 1:
            break
if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])