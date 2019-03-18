# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2019-01-30
#     DESC: 获取打分对象
#           对象为关键词-商品ID组成的Pair

import time
import pandas as pd
from log_handler.Log import log_handler
from foundation.utils import *
from traceback import format_exc as excp_trace
from data_client.goodssn_all_handler import GoodssnAllHandler
from foundation.file_path import *


# ES客户端获取商品详情
class SkuEsData(object):
    @classmethod
    def clear_all_goods_info(cls):
        for file_name in os.listdir(get_good_data_path()):
            try:
                time.sleep(0.5)
                log_handler.log.info("Remove good data file %s" % file_name)
                remove_good_data(file_name)
            except Exception as e:
                log_handler.log.info("Error when removing file good data file %s, %s" % (file_name, str(e)))

    @classmethod
    def build_all_goods_info(cls):
        features = [
            "goodsSn",
            "brandName",
            "categories",
            "discount",
            "shopPrice",
            "displayPrice",
            "stockFlag",
            "youtube",
            "totalFavoriteCount",
            "passTotalNum",
            "passAvgScore",
            "exposureSalesRate",
            "grossMargin",
            "exposureSalesVolume",
            "week2Sales",
            "week2SalesVolume",
            "dailyRate",
            "yesterdaySales",
            "goodsWebSpu"
        ]
        data_handler = GoodssnAllHandler(features)
        data_handler.build_goods()

    @classmethod
    def filter_all_goods_info(cls):
        column = ["pagecat", "goodssn"]
        df_test = pd.read_table(get_data_path() + SEARCH_TEST_LOCAL_FILE, sep='', header=None, names=column,
                                dtype={
                                    "pagecat": str,
                                    "goodssn": str
                                })
        unique_all_goodssn = list(set(df_test["goodssn"].values.tolist()))
        df_test = pd.DataFrame({"goodssn": unique_all_goodssn})
        rename_map = {
            "goodsSn":              "goodssn",
            "brandName":            "brand",
            "categories":           "categories",
            "discount":             "discount",
            "shopPrice":            "shop_price",
            "displayPrice":         "display_price",
            "stockFlag":            "stock_flag",
            "youtube":              "youtube",
            "totalFavoriteCount":   "favorite",
            "passTotalNum":         "score_num",
            "passAvgScore":         "avg_score",
            "exposureSalesRate":    "exposure_sales_rate",
            "grossMargin":          "gross_margin",
            "exposureSalesVolume":  "exposure_sales_volume",
            "week2Sales":           "week2sales",
            "week2SalesVolume":     "week2sales_volume",
            "dailyRate":            "daily_rate",
            "yesterdaySales":       "yesterday_sales"
        }

        file_counter = 0
        frames = []
        for file_name in os.listdir(get_good_data_path()):
            try:
                log_handler.log.info("Building goods feature batch %s" % file_counter)
                good_info_df = read_data_pickle("../good_data/" + file_name)
                good_info_df = good_info_df.rename(index=str, columns=rename_map)
                df_part = good_info_df.merge(df_test, left_on="goodssn", right_on="goodssn", how="inner")
                frames.append(df_part)
                file_counter += 1
            except Exception as e:
                log_handler.log.info(str(e))
        df_res = pd.concat(frames)
        pd.to_pickle(df_res, get_data_path()+ALL_GOODS_DETAIL)


def download_goods_detail():
    try:
        log_handler.log.info("----------------Building goods test info----------------")
        SkuEsData.clear_all_goods_info()
        SkuEsData.build_all_goods_info()
        SkuEsData.filter_all_goods_info()
        log_handler.log.info("----------------Finish downloading search test log----------------")
    except Exception:
        log_handler.log.info("----------------Error building goods test info----------------")
        log_handler.log.info(str(excp_trace()))
        raise Exception


if __name__ == '__main__':
    download_goods_detail()
