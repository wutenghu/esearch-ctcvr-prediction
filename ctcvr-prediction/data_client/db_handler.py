# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2018-12-13
#     DESC: 数据库接口
# from log_handler.Log import log_handler
import pymysql


class SQLWrapper(object):
    def __init__(self, host, port, database, user, password):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.conn = self.__get_connect()
        self.conn.autocommit(False)     # 自动commit设置为False，方便回滚

    def reconnect(self, host, port, database, user, password):
        self.conn.close()
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.conn = self.__get_connect()
        self.conn.autocommit(False)     # 自动commit设置为False，方便回滚

    def __get_connect(self):
        return pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password,
                               database=self.database, charset="utf8")

    def exec_query(self, sql):
        cur = self.conn.cursor()
        cur.execute(sql)
        result_list = cur.fetchall()
        return result_list

    def exec_insert(self, sql):
        cur = self.conn.cursor()
        cur.execute(sql)

    def escape_string(self, s):
        return self.conn.escape_string(s)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def destroy(self):
        self.conn.close()


class DbHandler(object):
    def __init__(self, db_dict):
        host = db_dict['host']
        port = db_dict['port']
        database = db_dict['database']
        user = db_dict['user']
        password = db_dict['password']
        self.reader = SQLWrapper(host=host, port=port, database=database, user=user, password=password)

    def reconnet(self, db_dict):
        host = db_dict['host']
        port = db_dict['port']
        database = db_dict['database']
        user = db_dict['user']
        password = db_dict['password']
        self.reader.reconnect(host=host, port=port, database=database, user=user, password=password)

    def reset_keyword_sku_score(self):
        sql = """update es_statistical.al_rerank_keyword_score_10002 set audit_status='I' 
                 where audit_id='ZYJ' and bts_policy='C'"""
        self.reader.exec_insert(sql)
        self.reader.commit()

    def algorithm_exist(self, algorithm_id):
        if isinstance(algorithm_id, int):
            sql = """select count(*) from es_statistical.admin_rerank_record where id='%s' 
                     and audit_status='A'""" % algorithm_id
            db_res = self.reader.exec_query(sql)
            self.reader.commit()
            if db_res[0][0] != 0:
                return True
            else:
                return False
        else:
            return False

    def update_algorithm_status(self, algorithm_id, status, desc, domain):
        # domain是外来参数
        domain = self.reader.escape_string(domain)
        sql = """update es_statistical.admin_rerank_record set task_status='%s', task_desc='%s', domain='%s' 
                 where id='%s'""" % (status, desc, domain, algorithm_id)
        self.reader.exec_query(sql)
        self.reader.commit()

    # 将商品的搜索场景得分写入数据库
    def insert_multi_keyword_sku_score(self, data_list):
        sql = """INSERT INTO es_statistical.al_rerank_keyword_score_10002
                (platform, keyword, goods_sn, bts_policy, score, audit_status, audit_id, audit_time) VALUES """
        for data in data_list:
            sql = sql + """('""" + """','""".join(data) + """'),"""
        sql = sql[:len(sql) - 1]
        sql = sql + " on DUPLICATE KEY UPDATE\
                     score=values(score),\
                     audit_status=values(audit_status),\
                     platform=values(platform),\
                     audit_id=values(audit_id),\
                     audit_time=values(audit_time);"

        self.reader.exec_insert(sql)
        self.reader.commit()
