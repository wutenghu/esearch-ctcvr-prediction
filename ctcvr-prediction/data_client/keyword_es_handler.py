# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2019-01-29
#     DESC: ES接口，接收关键词列表，返回商品详细列表

import elasticsearch
from elasticsearch import helpers
from foundation.utils import *


class KeywordEsHandler(object):
    def __init__(self, features, keyword_list):
        self.keyword_list = keyword_list
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
        self.result_dict["keyWord"] = []

    def search(self, index='GB'):
        for keyword in self.keyword_list:
            goodssn_set = set()
            es_search_options = self.set_search_optional(keyword)
            es_result = self.get_search_result(es_search_options, index)
            for item in es_result:
                try:
                    good_dict = item["_source"]
                    if good_dict["goodsSn"] not in goodssn_set:
                        goodssn_set.add(good_dict["goodsSn"])
                    else:
                        continue

                    for feature in self.features:
                        _ = good_dict[feature]
                    for feature in self.features:
                        self.result_dict[feature].append(good_dict[feature])
                    self.result_dict["keyWord"].append(keyword)
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

    def set_search_optional(self, keyword):
        # 检索选项
        es_search_options = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "bool": {
                                        "should": [
                                            {
                                                "match": {
                                                    "searchWords": {
                                                        "query": keyword,
                                                        "operator": "OR",
                                                        "prefix_length": 0,
                                                        "max_expansions": 50,
                                                        "minimum_should_match": "100%",
                                                        "fuzzy_transpositions": True,
                                                        "lenient": False,
                                                        "zero_terms_query": "NONE",
                                                        "boost": 0.1
                                                    }
                                                }
                                            },
                                            {
                                                "match": {
                                                    "subTitle": {
                                                        "query": keyword,
                                                        "operator": "OR",
                                                        "prefix_length": 0,
                                                        "max_expansions": 50,
                                                        "minimum_should_match": "100%",
                                                        "fuzzy_transpositions": True,
                                                        "lenient": False,
                                                        "zero_terms_query": "NONE",
                                                        "boost": 2
                                                    }
                                                }
                                            },
                                            {
                                                "match": {
                                                    "goodsModelWord": {
                                                        "query": keyword,
                                                        "operator": "OR",
                                                        "prefix_length": 0,
                                                        "max_expansions": 50,
                                                        "minimum_should_match": "100%",
                                                        "fuzzy_transpositions": True,
                                                        "lenient": False,
                                                        "zero_terms_query": "NONE",
                                                        "boost": 5
                                                    }
                                                }
                                            },
                                            {
                                                "match": {
                                                    "brandName": {
                                                        "query": keyword,
                                                        "operator": "OR",
                                                        "prefix_length": 0,
                                                        "max_expansions": 50,
                                                        "minimum_should_match": "100%",
                                                        "fuzzy_transpositions": True,
                                                        "lenient": False,
                                                        "zero_terms_query": "NONE",
                                                        "boost": 3
                                                    }
                                                }
                                            },
                                            {
                                                "nested": {
                                                    "query": {
                                                        "match": {
                                                            "skuAttrs.attrValue": {
                                                                "query": keyword,
                                                                "operator": "OR",
                                                                "prefix_length": 0,
                                                                "max_expansions": 50,
                                                                "minimum_should_match": "100%",
                                                                "fuzzy_transpositions": True,
                                                                "lenient": False,
                                                                "zero_terms_query": "NONE",
                                                                "boost": 1
                                                            }
                                                        }
                                                    },
                                                    "path": "skuAttrs",
                                                    "ignore_unmapped": False,
                                                    "score_mode": "sum",
                                                    "boost": 1
                                                }
                                            },
                                            {
                                                "nested": {
                                                    "query": {
                                                        "match": {
                                                            "skuDescAttrs.attrValue": {
                                                                "query": keyword,
                                                                "operator": "OR",
                                                                "prefix_length": 0,
                                                                "max_expansions": 50,
                                                                "minimum_should_match": "100%",
                                                                "fuzzy_transpositions": True,
                                                                "lenient": False,
                                                                "zero_terms_query": "NONE",
                                                                "boost": 1
                                                            }
                                                        }
                                                    },
                                                    "path": "skuDescAttrs",
                                                    "ignore_unmapped": False,
                                                    "score_mode": "sum",
                                                    "boost": 1
                                                }
                                            },
                                            {
                                                "match": {
                                                    "goodsTitle": {
                                                        "query": keyword,
                                                        "operator": "OR",
                                                        "prefix_length": 0,
                                                        "max_expansions": 50,
                                                        "minimum_should_match": "100%",
                                                        "fuzzy_transpositions": True,
                                                        "lenient": False,
                                                        "zero_terms_query": "NONE",
                                                        "boost": 5
                                                    }
                                                }
                                            }
                                        ],
                                        "disable_coord": False,
                                        "adjust_pure_negative": True,
                                        "boost": 1
                                    }
                                }
                            ],
                            "disable_coord": False,
                            "adjust_pure_negative": True,
                            "boost": 1
                        }
                    },
                    "functions": [
                        {
                            "filter": {
                                "match_all": {
                                    "boost": 1
                                }
                            },
                            "weight": 0.4,
                            "field_value_factor": {
                                "field": "exposureSalesVolume",
                                "factor": 1,
                                "missing": 0,
                                "modifier": "ln1p"
                            }
                        },
                        {
                            "filter": {
                                "match_all": {
                                    "boost": 1
                                }
                            },
                            "weight": 0.3,
                            "field_value_factor": {
                                "field": "exposureSalesRate",
                                "factor": 1,
                                "missing": 0,
                                "modifier": "sqrt"
                            }
                        },
                        {
                            "filter": {
                                "match_all": {
                                    "boost": 1
                                }
                            },
                            "weight": 0.1,
                            "field_value_factor": {
                                "field": "totalFavoriteCount",
                                "factor": 1,
                                "missing": 0,
                                "modifier": "log1p"
                            }
                        },
                        {
                            "filter": {
                                "range": {
                                    "firstUpTime": {
                                        "from": 1547521877,
                                        "to": None,
                                        "include_lower": True,
                                        "include_upper": True,
                                        "boost": 1
                                    }
                                }
                            },
                            "weight": 0.2
                        },
                        {
                            "filter": {
                                "match_all": {
                                    "boost": 1
                                }
                            },
                            "weight": 0.1,
                            "field_value_factor": {
                                "field": "grossMargin",
                                "factor": 1,
                                "missing": 0,
                                "modifier": "none"
                            }
                        },
                        {
                            "filter": {
                                "match_all": {
                                    "boost": 1
                                }
                            },
                            "weight": 0.2,
                            "field_value_factor": {
                                "field": "passAvgScore",
                                "factor": 1,
                                "missing": 0,
                                "modifier": "ln1p"
                            }
                        }
                    ],
                    "score_mode": "sum",
                    "boost_mode": "sum",
                    "max_boost": 3.4028235e+38,
                    "boost": 1
                }
            },
            "_source": {
                "includes": self.features,
                "excludes": []
            }
        }
        return es_search_options

    def build_goods(self):
        self.search(index='GB')
        return self.result_dict


if __name__ == '__main__':
    good_features = [
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
    data_client = KeywordEsHandler(good_features, ["xiaomi"])
    result_dict = data_client.build_goods()
    import pandas as pd
    df_features = pd.DataFrame(result_dict)
    df_features.to_csv(get_data_path()+"tmp.csv")