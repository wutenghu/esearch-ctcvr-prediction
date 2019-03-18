# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2019-01-29
#     DESC: 所有基础方法、变量、类

import pickle
import json
import os
import re
import sys
import csv
import codecs
import configparser
import shutil


reload(sys)
sys.setdefaultencoding('utf-8')


"""
# Windows环境
data_path = "F:/ctcvr-prediction/data/"
config_path = "F:/ctcvr-prediction/config/config.ini"
db_cache = "F:/ctcvr-prediction/db_cache/"
good_data_path = "F:/ctcvr-prediction/good_data/"
spark_cache = "F:/ctcvr-prediction/spark_cache/"
model_path = "F:/ctcvr-prediction/results/test/model/model_proto_buff"
log_path = "F:/ctcvr-prediction/log/"
"""

# linux环境
test_path = os.path.dirname(os.path.realpath(__file__))
path_items = test_path.split('/')
data_path = '/'.join(path_items[:-1] + ["data/"])
db_cache = '/'.join(path_items[:-1] + ["db_cache/"])
good_data_path = '/'.join(path_items[:-1] + ["good_data/"])
config_path = '/'.join(path_items[:-1] + ["config", "config_online.ini"])
spark_cache = '/'.join(path_items[:-1] + ["spark_cache/"])
model_path = '/'.join(path_items[:-1] + ["results/test/model/model_proto_buff"])
log_path = '/'.join(path_items[:-1] + ["log/"])


def list2chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def save2list(pname):
    csv_file = data_path + pname
    f = open(csv_file, 'w')
    return f


# ############################## 获取路径系列 ##############################
def get_data_path():
    return_path = data_path
    return return_path


def get_db_cache_path():
    return_path = db_cache
    return return_path


def get_spark_cache_path():
    return_path = spark_cache
    return return_path


def get_model_path():
    return_path = model_path
    return return_path


def get_good_data_path():
    return_path = good_data_path
    return return_path


# ############################## 小数据MAP_REDUCE系列 ##############################
def map_reduce(input_dict, input_key, input_value=1):
    if input_key in input_dict.keys():
        input_dict[input_key] += input_value
    else:
        input_dict[input_key] = input_value


def map_reduce_list(input_dict, input_key, input_value):
    if input_key in input_dict.keys():
        input_dict[input_key].append(input_value)
    else:
        input_dict[input_key] = [input_value]


def map_reduce_set(input_dict, input_key, input_value):
    if input_key in input_dict.keys():
        input_dict[input_key].add(input_value)
    else:
        input_dict[input_key] = {[input_value]}


# ############################## 帅气打印系列 ##############################
def print_dict(input_dict, filter=None):
    for key, value in input_dict.items():
        if filter is not None:
            if filter(key, value):
                print key, value
        else:
            print key, value


def print_list(input_list):
    for item in input_list:
        print item


def print_dict_list(input_dict, filter=None):
    for key, value in input_dict.items():
        if filter is not None:
            if filter(key, value):
                print key, len(value)
        else:
            print key, len(value)


# ############################## DB缓存读写 ##############################
def read_db_cache_json(file_name):
    file_name = db_cache + file_name
    f = open(file_name, 'r')
    data = json.loads(f.read())
    f.close()
    return data


def write_db_cache2json(data, file_name):
    file_name = db_cache + file_name
    out =  open(file_name, 'wb')
    out.write(bytes(json.dumps(data).encode()))
    out.close()


def read_db_cache_pickle(file_name):
    file_name = db_cache + file_name
    ancient_dict = pickle.load(open(file_name, "rb"))
    return ancient_dict


def write_db_cache2pickle(data, file_name):
    file_name = db_cache + file_name
    pickle.dump(data, open(file_name, "wb"))


def remove_db_cache(file_name):
    file_name = db_cache + file_name
    os.remove(file_name)


def remove_good_data(file_name):
    file_name = good_data_path + file_name
    os.remove(file_name)


def remove_data(file_name):
    file_name = data_path + file_name
    os.remove(file_name)


# ############################## DATA目录读写 ##############################
def read_data_json(file_name):
    file_name = data_path + file_name
    f = open(file_name, 'r')
    data = json.loads(f.read())
    f.close()
    return data


def write_data2json(data, file_name):
    file_name = data_path + file_name
    out =  open(file_name,'wb')
    out.write(bytes(json.dumps(data).encode()))
    out.close()


def read_data_pickle(file_name):
    file_name = data_path + file_name
    ancient_dict = pickle.load(open(file_name, "rb"))
    return ancient_dict


def write_data2pickle(data, file_name):
    file_name = data_path + file_name
    pickle.dump(data, open(file_name, "wb"))


def read_lines(file_name):
    file_name = data_path + file_name
    f = open(file_name)
    cache = f.readlines()
    f.close()
    return cache


def write_lines(data, file_name):
    file_name = data_path + file_name

    # open不支持utf8编码，which sucks
    f = codecs.open(file_name, "w", encoding='utf-8')
    counter = 0
    for line in data:
        # write line to output file
        try:
            f.write(line)
            # f.write("\n")
        except:
            print [line]
            counter += 1
    print counter
    f.close()


def read_csv_lines(file_name):
    csvfile = open(data_path + file_name)
    cache = []
    for row in csv.DictReader(csvfile):
        cache.append(row)
    csvfile.close()
    return cache


def read_csv_list(file_name):
    csvfile = open(data_path + file_name)
    cache = []
    for row in csv.reader(csvfile):
        cache.append(row)
    csvfile.close()
    return cache


# ############################## 文件夹操作 ##############################
def mkdir_ifnot_exist(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def rmdir_if_exist(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)


# ############################## SPARK目录读写 ##############################
def read_spark_lines(spark_dir, file_name):
    spark_cache_path = get_spark_cache_path()
    file_name = spark_cache_path + spark_dir + "/" + file_name
    # f = open(file_name)
    f = codecs.open(file_name, "r", encoding='utf-8')
    cache = f.readlines()
    f.close()
    return cache


def write_spark_lines(data, spark_dir, file_name):
    import traceback
    spark_cache_path = get_spark_cache_path()
    file_name = spark_cache_path + spark_dir + "/" + file_name

    # open不支持utf8编码，which sucks
    f = codecs.open(file_name, "w", encoding='utf-8')
    for line in data:
        # write line to output file
        try:
            f.write(line)
        except:
            print traceback.format_exc()
            print [line]
    f.close()


def intergrate_spark_files(file_name):
    file_list = []
    spark_cache_path = get_spark_cache_path()
    spark_dir = spark_cache_path + file_name
    pattern = "^part-\d+"
    for file_path in os.listdir(spark_dir):
        if re.match(pattern, file_path):
            file_list.append(file_path)

    great_list = []
    for file_path in file_list:
        sub_list = read_spark_lines(file_name, file_path)
        great_list.extend(sub_list)

    write_spark_lines(great_list, file_name, file_name + ".txt")


# ############################## 帅气打印 ##############################
def gen_grams(in_string):
    chars = in_string.strip().split()
    results = []
    for idx in range(len(chars)):
        unigram = chars[idx]
        bygram = "_".join(chars[idx:idx+2])
        trigram = "_".join(chars[idx:idx+3])
        results.extend([unigram, bygram, trigram])
    results = list(set(results))
    return results


# ############################## 帅气打印 ##############################
def dict_pretty_statistic(dict_in, bound_other=10):
    list_in = sorted(dict_in.items(), key=lambda d: d[1], reverse=True)
    list_sum = float(sum(value[1] for value in list_in))
    count = 0.0
    print "total", list_sum
    for value in list_in[0:bound_other]:
        fraction = float(value[1])/list_sum
        count += float(value[1])
        print value[0], value[1], "{0:.1f}%".format(fraction*100)
    other = (list_sum - count)/list_sum
    print "other", "{0:.1f}%".format(other*100)


# ############################## 读取数据库仓库配置 ##############################
def read_db_config(key_name="DataBase"):
    conf_file = open(config_path)
    config = configparser.ConfigParser()
    config.readfp(conf_file)
    db_dict = dict()
    host = config.get(key_name, "host")
    port = int(config.get(key_name, "port"))
    db_dict['host'] = host
    db_dict['port'] = port

    database = config.get(key_name, "database")
    user = config.get(key_name, "user")
    password = config.get(key_name, "password")
    db_dict['database'] = database
    db_dict['user'] = user
    db_dict['password'] = password
    return db_dict


def read_es_config(key_name="ESEARCH"):
    conf_file = open(config_path)
    config = configparser.ConfigParser()
    config.readfp(conf_file)
    # 读取配置
    cluster = config.get(key_name, "cluster")
    cluster = cluster.encode('utf-8')
    es_li = cluster.split(',')
    es_list = [i.split(':') for i in es_li]
    es_dict = {}
    es_lists = []
    import copy
    for i in es_list:
        es_dict['host'] = i[0]
        es_dict['port'] = i[1]
        es_lists.append(copy.copy(es_dict))
    return es_lists


def read_hadoop_config(key_name="HADOOP"):
    conf_file = open(config_path)
    config = configparser.ConfigParser()
    config.readfp(conf_file)
    # 读取配置
    host = config.get(key_name, "host")
    host = host.encode('utf-8')
    # 放置临时数据的目录
    return host


def read_redis_config(key_name='REDIS'):
    conf_file = open(config_path)
    config = configparser.ConfigParser()
    config.readfp(conf_file)
    # 读取配置
    cluster = config.get(key_name, "cluster")
    password = config.get(key_name, 'password')
    return cluster, password


if __name__ == '__main__':
    pass