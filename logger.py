# It is cubersome to set up a logger for every module. Instead, we can create a
# log utility module and import it when necessary.

import logging
from datetime import date
# You can customize the formatter according to your needs.
FORMATTER = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"

)
# You can change the file name if needed.
LOG_FILE = "python_tutorial_debug.log"


def get_stream_handler():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(FORMATTER)
    return stream_handler


def get_file_handler(name):
    LOG_FILE = str(date.today()) +name +'.log'
    file_handler = logging.FileHandler(filename=LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    logger.addHandler(get_stream_handler())
    logger.addHandler(get_file_handler(logger_name))

    return logger
def job():
  for i in range(5):
    log.info(f"Child thread: {i}")
    time.sleep(1)
def job2():
  for i in range(5):
    log.info(f"Child2 thread: {i}")
    time.sleep(1)
if __name__ == "__main__":
    log = get_logger(__name__)
    import threading
    import time


    # 建立一個子執行緒
    t = threading.Thread(target = job)
    t2 = threading.Thread(target = job2)
    # 執行該子執行緒
    t.start()
    t2.start()

    # 主執行緒繼續執行自己的工作
    for i in range(3):
        log.info(f"Main thread: {i}")
        time.sleep(1)

    # 等待 t 這個子執行緒結束
    t.join()
    t2.join()
    log.info("Done.")