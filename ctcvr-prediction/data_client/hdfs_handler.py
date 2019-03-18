# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2018-12-13
#     DESC: HDFS客户端，用于读写HDFS文件

import os
from log_handler.Log import log_handler
from hdfs.client import Client
from foundation.utils import read_hadoop_config


class HdfsHandler(object):
    @classmethod
    def get_hadoop_connection(cls, host):
        try:
            client = Client(host, root='/', timeout=10000)
            client.list('/')
        except Exception as e:
            try:
                log_handler.log.info('get query data error from hadoop 01 -----{}'.format(e))
                host = host.replace('01', '02')
                client = Client(host, root='/', timeout=10000)
                client.list('/')
            except Exception as e:
                try:
                    log_handler.log.info('get query data error from hadoop 02 -----{}'.format(e))
                    host = host.replace('02', '03')
                    client = Client(host, root='/', timeout=10000)
                    client.list('/')
                except Exception as e:
                    client = None
                    log_handler.log.info('get query data error from hadoop -----{}'.format(e))
        return client

    @classmethod
    def download_overwrite_file(cls, hdfs_file_path, local_file_path, key_name="HADOOP"):
        host = read_hadoop_config(key_name)
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        client = cls.get_hadoop_connection(host)
        client.download(hdfs_file_path, local_file_path)

    @classmethod
    def list_dir(cls, hdfs_dir_path, key_name="HADOOP"):
        host = read_hadoop_config(key_name)
        client = cls.get_hadoop_connection(host)
        file_list = client.list(hdfs_dir_path)
        return file_list

    @classmethod
    def upload_overwrite_file(cls, hdfs_file_path, local_file_path, key_name="HADOOP"):
        file_ptr = open(local_file_path, "rb+")
        data = file_ptr.read(-1)
        host = read_hadoop_config(key_name)
        client = cls.get_hadoop_connection(host)
        client.write(hdfs_file_path, data, True)
        file_ptr.close()

    @classmethod
    def check_dir(cls, path, key_name):
        host = read_hadoop_config(key_name)
        client = cls.get_hadoop_connection(host)
        client.makedirs(path)

    @classmethod
    def upload_overwrite_dir(cls, hdfs_dir, local_dir, key_name="HADOOP"):
        if os.path.isdir(local_dir):
            cls.check_dir(hdfs_dir, key_name)
            nodes = os.listdir(local_dir)
            for i in range(0, len(nodes)):
                path = os.path.join(local_dir, nodes[i])
                remote_dir_new = hdfs_dir + nodes[i] if hdfs_dir[-1] == "/" else hdfs_dir + "/" + nodes[i]
                if os.path.isfile(path):
                    cls.upload_overwrite_file(remote_dir_new, path)
                elif os.path.isdir(path):
                    cls.check_dir(remote_dir_new, key_name)
                    cls.upload_overwrite_dir(remote_dir_new, path, key_name)


if __name__ == '__main__':
    pass
