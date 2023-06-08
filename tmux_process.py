import subprocess
import sys
import os
import time


def tmux_exec():
    script2_path = "/home/allenkuo/tw_pt_stock_future/trade_signal.py"
    script1_path = "/home/allenkuo/tw_pt_stock_future/trade.py"
    script_paths = [script1_path, script2_path]
    env_name = "pair_trading"
    # 創建一個tmux窗口
    
    time.sleep(2)
    trading_pair = [
        ["2885", "2887"],
        ["2352", "3481"],
        ["2603", "2887"],
        ["2353", "2887"],
        ["2603", "5880"],
        ["2352", "2885"],
        ["2352", "2887"],
        ["2353", "2885"],
        ["2352", "2603"],
        ["2610", "2618"],
    ]
    count = 0
    for pair in trading_pair[:]:
        subprocess.run(["tmux", "new-session", "-d", "-s", f"{count}_{pair[0]}_{pair[1]}_pt"])
        for i, script_path in enumerate(script_paths):
            if i == 0:
                # 第一個窗格，不需要拆分畫面
                subprocess.run(
                    ["tmux", "send-keys", f"conda activate {env_name}\n", "C-m"]
                )
                subprocess.run(
                    [
                        "tmux",
                        "send-keys",
                        f"python {script_path} {pair[0]} {pair[1]}\n",
                        "C-m",
                    ]
                )
                time.sleep(2)
            elif i == 1:
                subprocess.run(["tmux", "split-window", "-h"])
                subprocess.run(["tmux", "select-layout", "even-horizontal"])
                subprocess.run(
                    ["tmux", "send-keys", f"conda activate {env_name}\n", "C-m"]
                )
                subprocess.run(
                    [
                        "tmux",
                        "send-keys",
                        f"python {script_path} {pair[0]} {pair[1]}\n",
                        "C-m",
                    ]
                )
                time.sleep(2)
        os.system(f"sleep 5 && tmux send-keys -t {count}_{pair[0]}_{pair[1]}_pt C-b d")
        count += 1
        time.sleep(2)


if __name__ == "__main__":
    tmux_exec()