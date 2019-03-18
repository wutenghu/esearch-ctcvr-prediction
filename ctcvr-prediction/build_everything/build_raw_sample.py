# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2018-12-13
#     DESC: 获取训练数据关联商品详情
#           运行时间大约1小时

import time
import pandas as pd
from traceback import format_exc as excp_trace
from log_handler.Log import log_handler
from foundation.utils import *
from data_client.hdfs_handler import HdfsHandler
from foundation.file_path import *


# 处理HDFS日志
class HdfsTrainLog(object):
    @classmethod
    def clear_all_data_info(cls):
        for file_name in os.listdir(get_data_path()):
            try:
                time.sleep(0.5)
                log_handler.log.info("Remove data file %s" % file_name)
                remove_data(file_name)
            except Exception as e:
                log_handler.log.info("Error when removing data file %s, %s" % (file_name, str(e)))

    @classmethod
    def download_traning_from_hdfs(cls):
        file_names = HdfsHandler.list_dir(SEARCH_TRAIN_HDFS_DIR)
        for file_name in file_names:
            if re.match("\d{6}_\d", file_name):
                HdfsHandler.download_overwrite_file(SEARCH_TRAIN_HDFS_DIR+file_name, get_data_path()+file_name)
        file_lines = []
        for file_name in file_names:
            if re.match("\d{6}_\d", file_name):
                lines = read_lines(file_name)
                file_lines.extend(lines)
        write_lines(file_lines, SEARCH_TRAIN_LOCAL_FILE)

    @classmethod
    def convert_hdfs_to_list(cls):
        column = ["catid", "goodssn", "expose_total", "click_total", "purchase_total", "clicked", "purchased"]
        hdf = pd.read_table(get_data_path() + SEARCH_TRAIN_LOCAL_FILE, sep='', header=None, names=column,
                            dtype={
                                "catid": str,
                                "goodssn": str,
                                "expose_total": float,
                                "click_total": float,
                                "purchase_total": float,
                                "exposed": float,
                                "clicked": float,
                                "purchased": float
                            })
        items = hdf.values.tolist()
        unq_pagecat_list = list(set([item[0] for item in items if item[0].strip()]))
        write_data2pickle(unq_pagecat_list, UNIQUE_KEYWORD_LIST)


# 处理HDFS日志
class HdfsTestLog(object):
    @classmethod
    def download_testing_from_hdfs(cls):
        file_names = HdfsHandler.list_dir(SEARCH_TEST_HDFS_DIR)
        for file_name in file_names:
            if re.match("\d{6}_\d", file_name):
                HdfsHandler.download_overwrite_file(SEARCH_TEST_HDFS_DIR+file_name, get_data_path()+file_name)
        file_lines = []
        for file_name in file_names:
            if re.match("\d{6}_\d", file_name):
                lines = read_lines(file_name)
                file_lines.extend(lines)
        write_lines(file_lines, SEARCH_TEST_LOCAL_FILE)

    @classmethod
    def convert_hdfs_to_list(cls):
        column = ["keyword", "goodssn"]
        hdf = pd.read_table(get_data_path() + SEARCH_TEST_LOCAL_FILE, sep='', header=None, names=column,
                            dtype={
                                "keyword": str,
                                "goodssn": str
                            })
        items = hdf.values.tolist()
        all_goodssn_list = list(set([item[1] for item in items if item[1].strip()]))
        write_data2pickle(all_goodssn_list, ALL_GOODSSN_LIST)


def build_raw_sample():
    try:
        log_handler.log.info("----------------Downloading raw samples for training----------------")
        HdfsTrainLog.clear_all_data_info()
        HdfsTrainLog.download_traning_from_hdfs()
        HdfsTrainLog.convert_hdfs_to_list()
        log_handler.log.info("----------------Finish downloading raw samples----------------")
        time.sleep(5)

        log_handler.log.info("----------------Downloading raw samples for testing----------------")
        HdfsTestLog.download_testing_from_hdfs()
        HdfsTestLog.convert_hdfs_to_list()
        log_handler.log.info("----------------Finish downloading raw samples----------------")
        time.sleep(5)
    except Exception:
        log_handler.log.info("----------------Error building raw samples----------------")
        log_handler.log.info(str(excp_trace()))
        raise Exception


if __name__ == '__main__':
    build_raw_sample()
