# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2019-01-29
#     DESC: ES接口，接收商品ID列表，返回商品详细列表

import elasticsearch
from elasticsearch import helpers
from foundation.utils import *


class GoodssnEsHandler(object):
    def __init__(self, features, goodssn_list):
        self.goodssn_list = goodssn_list
        self.features = features
        self.result_dict = dict()
        self.init_result_dict()
        es_servers = read_es_config("ESEARCH")
        self.es_client = elasticsearch.Elasticsearch(
            hosts=es_servers
        )

    def init_result_dict(self):
        for feature in self.features:
            self.result_dict[feature] = []

    def search(self, index='GB'):
        for goodssn in self.goodssn_list:
            try:
                es_search_options = self.set_search_optional(goodssn)
                es_result = self.get_search_result(es_search_options, index)
                for item in es_result:
                    good_dict = item["_source"]
                    for feature in self.features:
                        _ = good_dict[feature]
                    for feature in self.features:
                        self.result_dict[feature].append(good_dict[feature])
                    break
            except:
                pass

    def get_search_result(self, es_search_options, index2, scroll='1m', preserve_order=True):
        es_result = helpers.scan(
            client=self.es_client,
            query=es_search_options,
            scroll=scroll,
            index=index2,
            doc_type='sku',
            preserve_order=preserve_order
        )
        return es_result

    def set_search_optional(self, goodsn):
        # 检索选项
        es_search_options = {
            "query": {
                "term": {
                    "goodsSn": {
                        "value": goodsn
                    }
                }
            },
            "_source": self.features
        }
        return es_search_options

    def build_goods(self):
        self.search(index='GB')
        return self.result_dict
