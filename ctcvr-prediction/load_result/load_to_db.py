# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2019-01-30
#     DESC: 商品搜索场景评分结果写入数据库

import time
import pandas as pd
from log_handler.Log import log_handler
from foundation.utils import *
from foundation.file_path import *
from data_client.db_handler import DbHandler
from traceback import format_exc as excp_trace

INSERT_BATCH_SIZE = 100000


class LoadDb(object):
    ALG_STATUS_MAP = {
        1: "GB CTCVR RUNNING",
        2: "GB CTCVR SUCCESS",
        3: "GB CTCVR FAILURE"
    }

    def __init__(self):
        self.final_result = None

    @staticmethod
    def prepare_string(s):
        s = s.strip()
        return re.escape(s)

    @staticmethod
    def score_decoration(score):
        score = score*10.0
        return score

    def read_final_result(self, agent, bts_policy):
        agent = self.prepare_string(agent)
        bts_policy = self.prepare_string(bts_policy)

        self.final_result = read_data_pickle(FINAL_RESULT)
        self.final_result = pd.DataFrame(self.final_result)
        self.final_result = self.final_result[["goodssn", "keyword", "overall_prob"]]

        self.final_result["overall_prob"] = self.final_result["overall_prob"].apply(self.score_decoration)
        self.final_result["overall_prob"] = self.final_result["overall_prob"].astype(str)
        self.final_result["goodssn"] = self.final_result["goodssn"].astype(str)
        self.final_result["keyword"] = self.final_result["keyword"].astype(str)

        self.final_result["goodssn"] = self.final_result["goodssn"].apply(self.prepare_string)
        self.final_result["keyword"] = self.final_result["keyword"].apply(self.prepare_string)

        self.final_result["platform"] = agent
        self.final_result["bts_policy"] = bts_policy
        self.final_result["audit_status"] = "A"
        self.final_result["audit_id"] = "ZYJ"
        self.final_result["audit_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 重新整理列的顺序
        self.final_result = self.final_result[["platform", "keyword", "goodssn", "bts_policy", "overall_prob",
                                               "audit_status", "audit_id", "audit_time"]]
        self.final_result = self.final_result.values.tolist()
        # print_list(self.final_result)

    @staticmethod
    def reset_sku_keyword_score():
        db_config = read_db_config("ETL")
        db_client = DbHandler(db_config)
        db_client.reset_keyword_sku_score()

    @staticmethod
    def algorithm_exist(algorithm_id):
        db_config = read_db_config("ETL")
        db_client = DbHandler(db_config)
        return db_client.algorithm_exist(algorithm_id)

    def update_algorithm_status(self, algorithm_id, status, domain):
        desc = self.ALG_STATUS_MAP[status]
        db_config = read_db_config("ETL")
        db_client = DbHandler(db_config)
        db_client.update_algorithm_status(algorithm_id, status, desc, domain)

    def load_sku_keyword_score(self):
        for result_chunk in list2chunks(self.final_result, INSERT_BATCH_SIZE):
            db_config = read_db_config("ETL")
            db_client = DbHandler(db_config)
            db_client.insert_multi_keyword_sku_score(result_chunk)
            log_handler.log.info("%s rows inserted(update) into database" % len(result_chunk))


def load_to_db(agent, bts_policy):
    try:
        log_handler.log.info("----------------Loading search sku score----------------")
        handler = LoadDb()
        handler.read_final_result(agent, bts_policy)
        handler.load_sku_keyword_score()
        log_handler.log.info("----------------Done loading search sku score----------------")
    except Exception:
        log_handler.log.info("----------------Error loading search sku score----------------")
        log_handler.log.info(str(excp_trace()))
        raise Exception


def check_algorithm(algorithm_id):
    try:
        log_handler.log.info("----------------Check if algorithm with id:%s exist----------------" % algorithm_id)
        handler = LoadDb()
        log_handler.log.info("----------------Done checking algorithm existence----------------")
        return handler.algorithm_exist(algorithm_id)
    except Exception:
        log_handler.log.info("----------------Error checking algorithm existence----------------")
        log_handler.log.info(str(excp_trace()))
        raise Exception


def update_algorithm(algorithm_id, status, domain):
    try:
        log_handler.log.info("----------------Update algorithm id:%s with status %s----------------"
                             % (algorithm_id, status))
        handler = LoadDb()
        log_handler.log.info("----------------Done updating algorithm status----------------")
        return handler.update_algorithm_status(algorithm_id, status, domain)
    except Exception:
        log_handler.log.info("----------------Error updating algorithm status----------------")
        log_handler.log.info(str(excp_trace()))
        raise Exception


if __name__ == '__main__':
    load_to_db("pc", "C")
