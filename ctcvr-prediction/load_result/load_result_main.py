# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2018-12-28
#     DESC: 计算结果总过程

import time
import pandas as pd
from traceback import format_exc as excp_trace
from foundation.utils import *
from foundation.file_path import *
from model.esmm_model import get_trained_model
from log_handler.Log import log_handler
from load_result.load_to_db import LoadDb


class LoadResultMain(object):
    def __init__(self):
        self.keyword_list = []
        self.model = None

    def run(self, agent, bts_policy):

        log_handler.log.info("Reload Esmm ctcvr model")
        self.model = get_trained_model()

        log_handler.log.info("Reset all ctcvr score")
        handler = LoadDb()
        handler.reset_sku_keyword_score()

        log_handler.log.info("Load sku feature and sku-keyword pair")

        df_test_sample = pd.read_csv(get_data_path() + FEATURE_TEST_SOURCE, dtype={
            "pagecat": int,
            "brand": int,
            "cat": int,
            "cat1": int
        })

        log_handler.log.info("Begin scoring chunks of sku-keyword pairs")
        for df_chunk in list2chunks(df_test_sample, 500000):
            try:
                self.model.predict(df_chunk, FINAL_RESULT)
                log_handler.log.info("Loading score to DB")
                handler = LoadDb()
                handler.read_final_result(agent, bts_policy)
                handler.load_sku_keyword_score()
            except Exception as e:
                log_handler.log.info("Error scoring sample chunk")
                log_handler.log.info(str(excp_trace()))
        self.model.clear_all()


def load_result_main(agent, bts_policy):
    try:
        log_handler.log.info("----------------Processing batch result----------------")
        task_handler = LoadResultMain()
        task_handler.run(agent, bts_policy)
        log_handler.log.info("----------------Finish processing batch result----------------")
        time.sleep(5)
    except Exception:
        log_handler.log.info("----------------Error building batch result----------------")
        log_handler.log.info(str(excp_trace()))
        raise Exception


if __name__ == '__main__':
    load_result_main("pc", "C")
