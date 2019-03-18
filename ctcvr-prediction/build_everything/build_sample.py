# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2018-12-28
#     DESC: 特征工程，产生最后的训练样本
#           结果保存在data目录下:
#           1.brand2id.pkl
#           2.cat2id.pkl
#           3.cat12id.pkl
#           4.id2brand.pkl
#           5.id2cat.pkl
#           6.id2cat1.pkl
#           7.id2pagecat.pkl
#           8.pagecat2id.pkl
#           9.features.csv

import time
import pandas as pd
from traceback import format_exc as excp_trace
from foundation.utils import *
from log_handler.Log import log_handler
from foundation.arithmetic import *
from foundation.file_path import *


class BuildSample(object):
    # 读取文件/data/orig_goods_detail.pkl，/data/pagecat_list.pkl
    # 创建ID映射表，作为嵌入层的寻址标准
    # 输出文件保存为/data/xxx2id.pkl，或者/data/id2xxx.pkl
    @classmethod
    def build_id_dict(cls):
        good_info = pd.read_pickle(get_data_path()+ALL_GOODS_DETAIL)

        pagecat2id = {"unknown": 0}
        cat2id = {"unknown": 0}
        cat12id = {"unknown": 0}
        brand2id = {"unknown": 0}
        id2pagecat = {0: "unknown"}
        id2cat = {0: "unknown"}
        id2cat1 = {0: "unknown"}
        id2brand = {0: "unknown"}

        pagecat_counter = 1
        cat_counter = 1
        cat1_counter = 1
        brand_counter = 1

        pagecats = read_data_pickle(UNIQUE_KEYWORD_LIST)
        brands = good_info["brand"]
        categories = good_info["categories"]

        for brand in brands:
            if brand not in brand2id.keys() and brand.strip():
                brand2id[brand] = brand_counter
                id2brand[brand_counter] = brand
                brand_counter += 1

        for category in categories:
            if len(category) < 2:
                continue

            cat = category[0]["catName"]
            cat1 = category[1]["catName"]

            if cat not in cat2id.keys() and cat.strip():
                cat2id[cat] = cat_counter
                id2cat[cat_counter] = cat
                cat_counter += 1

            if cat1 not in cat12id.keys() and cat1.strip():
                cat12id[cat1] = cat1_counter
                id2cat1[cat1_counter] = cat1
                cat1_counter += 1

        for pagecat in pagecats:
            if pagecat not in pagecat2id.keys() and pagecat.strip():
                pagecat2id[pagecat] = pagecat_counter
                id2pagecat[pagecat_counter] = pagecat
                pagecat_counter += 1

        write_data2pickle(brand2id, BRAND_TO_ID)
        write_data2pickle(cat2id, CAT_TO_ID)
        write_data2pickle(cat12id, CAT1_TO_ID)
        write_data2pickle(pagecat2id, PAGECAT_TO_ID)
        write_data2pickle(id2pagecat, ID_TO_PAGECAT)
        write_data2pickle(id2brand, ID_TO_BRAND)
        write_data2pickle(id2cat, ID_TO_CAT)
        write_data2pickle(id2cat1, ID_TO_CAT1)

    # 获取单个样本的三级分类ID
    @classmethod
    def extract_cat_id(cls, category, cat2id):
        if len(category) > 0:
            cat_name = category[0]["catName"]
            cat_id = cat2id.get(cat_name, 0)
        else:
            cat_id = cat2id["unknown"]
        return cat_id

    # 获取单个样本的二级分类ID
    @classmethod
    def extract_cat1_id(cls, category, cat12id):
        if len(category) > 1:
            cat1_name = category[1]["catName"]
            cat1_id = cat12id.get(cat1_name, 0)
        else:
            cat1_id = cat12id["unknown"]
        return cat1_id

    # 获取单个样本的品牌ID
    @classmethod
    def extract_brand_id(cls, brand, brand2id):
        return brand2id.get(brand, 0)

    # 获取单个样本的关键词ID
    @classmethod
    def extract_pagecat(cls, pagecat, pagecat2id):
        return pagecat2id.get(pagecat, 0)

    # 获取单个样本的CVR或CTR标记
    @classmethod
    def split_ctcvr_value(cls, ctcvr):
        if ctcvr >= 0.01:
            return 1.0
        else:
            return 0

    @classmethod
    def youtube_normalization(cls, url):
        if url:
            return 1.0
        else:
            return 0.0

    @classmethod
    def stock_flag_normalization(cls, stock_flag):
        if stock_flag >= 1:
            return 1.0
        else:
            return 0.0

    @classmethod
    def make_good_feature(cls):
        brand2id = read_data_pickle(BRAND_TO_ID)
        cat2id = read_data_pickle(CAT_TO_ID)
        cat12id = read_data_pickle(CAT1_TO_ID)

        # 预处理商品特征
        df = pd.read_pickle(get_data_path() + ALL_GOODS_DETAIL)

        df["brand"] = df["brand"].apply(cls.extract_brand_id, args=(brand2id,))
        df["cat"] = df["categories"].apply(cls.extract_cat_id, args=(cat2id,))
        df["cat1"] = df["categories"].apply(cls.extract_cat1_id, args=(cat12id,))
        df["youtube"] = df["youtube"].apply(cls.youtube_normalization)
        df["stock_flag"] = df["stock_flag"].apply(cls.stock_flag_normalization)

        df["discount"] = df["discount"].apply(flat_head, args=(0, 100))
        df["discount"] = df["discount"].apply(max_min_normalization, args=(0, 100))
        df["shop_price"] = df["shop_price"].apply(price_bucket)
        df["display_price"] = df["display_price"].apply(price_bucket)
        df["avg_score"] = df["avg_score"].apply(score_normalization)

        score_num_iqr = inter_qr(df["score_num"], 2.5, 0.95, 0.98)
        df["score_num"] = df["score_num"].apply(box_normalization, args=(df["score_num"].min(), score_num_iqr))

        favorite_iqr = inter_qr(df["favorite"], 2.5, 0.95, 0.98)
        df["favorite"] = df["favorite"].apply(box_normalization, args=(df["favorite"].min(), favorite_iqr))

        exposure_sales_rate_iqr = inter_qr(df["exposure_sales_rate"], 2.5, 0.95, 0.98)
        df["exposure_sales_rate"] = df["exposure_sales_rate"].apply(box_normalization, args=(df["exposure_sales_rate"].min(), exposure_sales_rate_iqr))

        exposure_sales_volume_iqr = inter_qr(df["exposure_sales_volume"], 2.5, 0.95, 0.98)
        df["exposure_sales_volume"] = df["exposure_sales_volume"].apply(box_normalization, args=(df["exposure_sales_volume"].min(), exposure_sales_volume_iqr))

        week2sales_iqr = inter_qr(df["week2sales"], 2.5, 0.95, 0.99)
        df["week2sales"] = df["week2sales"].apply(box_normalization, args=(df["week2sales"].min(), week2sales_iqr))

        week2sales_volume_iqr = inter_qr(df["week2sales_volume"], 2.5, 0.95, 0.98)
        df["week2sales_volume"] = df["week2sales_volume"].apply(box_normalization, args=(df["week2sales_volume"].min(), week2sales_volume_iqr))

        daily_rate_iqr = inter_qr(df["daily_rate"], 2.5, 0.95, 0.99)
        df["daily_rate"] = df["daily_rate"].apply(box_normalization, args=(df["daily_rate"].min(), daily_rate_iqr))

        yesterday_sales_iqr = inter_qr(df["yesterday_sales"], 2.5, 0.95, 0.99)
        df["yesterday_sales"] = df["yesterday_sales"].apply(box_normalization, args=(df["yesterday_sales"].min(), yesterday_sales_iqr))

        df["gross_margin"] = df["gross_margin"].apply(gmatan_normalization)

        df = df.drop(["categories"], axis=1)

        # 预处理后统一去除异常数据
        for field_name in ["discount", "shop_price", "display_price", "avg_score", "score_num", "favorite",
                           "exposure_sales_rate", "exposure_sales_volume", "week2sales", "week2sales_volume",
                           "daily_rate", "yesterday_sales", "gross_margin"]:
            df = df.loc[np.isnan(df[field_name])==False]

        pd.to_pickle(df, get_data_path()+FEATURE_ALL_SOURCE)

    @classmethod
    def make_training_sample(cls):
        # 预处理用户特征
        df = read_data_pickle(FEATURE_ALL_SOURCE)
        pagecat2id = read_data_pickle(PAGECAT_TO_ID)
        column = ["pagecat", "goodssn", "expose_total", "click_total", "purchase_total", "clicked", "purchased"]
        df_user = pd.read_table(get_data_path() + SEARCH_TRAIN_LOCAL_FILE, sep='', header=None, names=column,
                                dtype={
                                    "pagecat": str,
                                    "goodssn": str,
                                    "expose_total": float,
                                    "click_total": float,
                                    "purchase_total": float,
                                    "clicked": float,
                                    "purchased": float
                                })
        df_user["keyword"] = df_user["pagecat"]
        df_user["click_rate"] = df_user["clicked"]
        df_user["purchase_rate"] = df_user["purchased"]
        df_user["pagecat"] = df_user["pagecat"].apply(cls.extract_pagecat, args=(pagecat2id,))
        df_user["clicked"] = df_user["clicked"].apply(cls.split_ctcvr_value)
        df_user["purchased"] = df_user["purchased"].apply(cls.split_ctcvr_value)

        # 用户特征与商品特征融合
        df_res = df_user.merge(df, left_on="goodssn", right_on="goodssn", how="inner")

        # 针对昂贵商品做重新采样
        df_purchase = df_res.loc[df_res["purchased"] > 0.5]
        df_no_purchase = df_res.loc[df_res["purchased"] < 0.5]
        df_purchase = df_purchase.append([df_purchase.loc[df_purchase["display_price"] == 0.75]]*9, ignore_index=True)
        df_purchase = df_purchase.append([df_purchase.loc[df_purchase["display_price"] == 0.875]]*9, ignore_index=True)
        df_purchase = df_purchase.append([df_purchase.loc[df_purchase["display_price"] == 1.0]]*14, ignore_index=True)

        df_res = pd.concat([df_purchase, df_no_purchase])
        df_res.to_csv(get_data_path() + FEATURE_TRAIN_SOURCE)

    @classmethod
    def make_testing_sample(cls):
        df = read_data_pickle(FEATURE_ALL_SOURCE)
        pagecat2id = read_data_pickle(PAGECAT_TO_ID)
        column = ["pagecat", "goodssn"]
        df_user = pd.read_table(get_data_path() + SEARCH_TEST_LOCAL_FILE, sep='', header=None, names=column,
                                dtype={
                                    "pagecat": str,
                                    "goodssn": str
                                })
        df_user["keyword"] = df_user["pagecat"]
        df_user["pagecat"] = df_user["pagecat"].apply(cls.extract_pagecat, args=(pagecat2id,))
        df_res = df_user.merge(df, left_on="goodssn", right_on="goodssn", how="inner")
        df_res.to_csv(get_data_path() + FEATURE_TEST_SOURCE)


def build_sample():
    try:
        log_handler.log.info("----------------Building id dictionaries----------------")
        BuildSample.build_id_dict()
        log_handler.log.info("----------------Finish building id dictionaries----------------")
        time.sleep(5)

        log_handler.log.info("----------------Building all goods features----------------")
        BuildSample.make_good_feature()
        log_handler.log.info("----------------Finish building all goods features----------------")
        time.sleep(5)

        log_handler.log.info("----------------Building training sample----------------")
        BuildSample.make_training_sample()
        log_handler.log.info("----------------Finish building all training sample----------------")
        time.sleep(5)

        log_handler.log.info("----------------Building testing sample----------------")
        BuildSample.make_testing_sample()
        log_handler.log.info("----------------Finish building testing sample----------------")
        time.sleep(5)

    except Exception:
        log_handler.log.info("----------------Error building sample----------------")
        log_handler.log.info(str(excp_trace()))
        raise Exception


if __name__ == '__main__':
    build_sample()
