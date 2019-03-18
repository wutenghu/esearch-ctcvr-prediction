# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2019-02-14
#     DESC: 通知redis，数据已备好
import redis
import time
from foundation.utils import *
from log_handler.Log import log_handler


class InformRedis(object):
    @staticmethod
    def get_redis_conn():
        cluster, password = read_redis_config("REDIS")

        # 测试环境有密码，线上没有密码
        con = None
        c1 = cluster.split(',')
        for i in c1:
            j = i.split(':')
            r = redis.Redis(host=j[0], port=int(j[1]), password=password)
            try:
                r.set('name', 'zhang')  # 测试连接
                con = r
                break
            except Exception as e:
                pass
        try:
            log_handler.log.info("Connection successful: {}".format(con['name']))
        except Exception as e:
            log_handler.log.info('Redis config is Error,is {}'.format(e))
        return con


def inform_redis_msg(channel, message):
    conn = InformRedis.get_redis_conn()
    try:
        conn.publish(channel, message)
        time.sleep(4)
        log_handler.log.info('redis channel is {} msg is {}'.format(channel, message))
        return True
    except:
        log_handler.log.info('redis publish fail')
        return False


if __name__ == '__main__':
    # inform_redis_msg()
    pass