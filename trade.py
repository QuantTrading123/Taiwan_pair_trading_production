import asyncio
from module.tw_pt_signal import TF_pair_trading
import shioaji as sj
from config import Pair_Trading_Config
from stock_config import Stock_Config
import os
import sys
async def main(ref_symbol,target_symbol):
    api = sj.Shioaji()
    from credentials import api_key, api_secret
    api.login(api_key, api_secret)
    configs = Pair_Trading_Config(ref_symbol,target_symbol)
    s_to_f_config = Stock_Config
    path = f"./Trading_Log/{configs.REFERENCE_SYMBOL}{configs.TARGET_SYMBOL}_tick/_{configs.REFERENCE_SYMBOL}{configs.TARGET_SYMBOL}_{int(configs.MA_WINDOW_SIZE/60)}length_Trading_log/"
    # 如果路徑不存在，則建立一個新的路徑
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
    tw_pair_trading = TF_pair_trading(api, api, configs,s_to_f_config,path)
    await tw_pair_trading.execute()
if __name__ == '__main__':
    asyncio.run(main(sys.argv[1],sys.argv[2]))