# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2018-12-13
#     DESC: 通用LOG句柄

import configparser
import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
from foundation.utils import log_path, config_path


level_dict = {
    'logging.CRITICAL': 50,
    'logging.FATAL': "CRITICAL",
    'logging.ERROR': 40,
    'logging.WARNING': 30,
    'logging.WARN': 'WARNING',
    'logging.INFO': 20,
    'logging.DEBUG': 10,
    'logging.NOTSET': 0
}


class Log:
    def __init__(self, name):
        log_filename = log_path + name + ".txt"
        cf = configparser.RawConfigParser()
        cf.read(config_path)

        # 通用配置
        config_name = name
        level = cf.get(config_name, 'level')
        # format = cf.get('logging', 'format')
        datefmt = cf.get(config_name, 'datefmt')
        filemode = cf.get(config_name, 'filemode')
        logging.basicConfig(level=level_dict.get(level, logging.INFO),
                            format='[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s',
                            datefmt=datefmt,
                            filemode=filemode
                            )

        self.log = logging.getLogger(name)
        if len(self.log.handlers) == 0:
            '''
            这里请注意，在运行定时框架时，常常发现LOG越打越多的问题
            原因很简单，上面那个logging.getLogger获取的是logging资源池
            中的一个资源的指针。任何一个python进程对应logging资源池只有一个
            定时任务会重复执行本类中的__init__方法。如果第一次执行时已经在
            logging资源池中添加了句柄资源，第二次若不判断该资源是否已经添加了
            句柄资源继续添加句柄，则会重复添加。这样一个logging资源所有的句柄
            会越来越多最终导致log越打越多
            '''

            '''
            # 一般的文件句柄
            handler = logging.FileHandler(log_filename)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.log.addHandler(handler)
            '''

            '''
            # 标准输出句柄
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.log.addHandler(ch)
            '''

            # 时间循环文件句柄
            rotating_handler = TimedRotatingFileHandler(filename=log_filename, when="D", interval=1,
                                                        backupCount=0, encoding="utf8")
            rotating_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            rotating_handler.setFormatter(formatter)
            self.log.addHandler(rotating_handler)


log_handler = Log("ctcvr-prediction")


if __name__ == '__main__':
    log_handler = Log("ctcvr-prediction")
    log_handler.log.info("All work and no play makes Jack a dull boy")
