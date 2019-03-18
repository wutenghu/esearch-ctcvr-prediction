# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2018-12-13
#     DESC: 训练ESMM模型

import os
import time
import numpy as np
import tensorflow as tf
import pandas as pd
from log_handler.Log import log_handler
from traceback import format_exc as excp_trace
from sklearn.metrics import roc_auc_score
from foundation.utils import get_data_path, read_data_pickle, rmdir_if_exist
from foundation.file_path import *


class EsmmModel(object):

    def __init__(self,
                 cat_names_feature_size=16,        # 1173
                 cat_names_embedding_size=3,       # 8

                 cat1_names_feature_size=16,       # 241
                 cat1_names_embedding_size=3,      # 4

                 brand_names_feature_size=16,      # 3095
                 brand_names_embedding_size=3,     # 16

                 pagecat_names_feature_size=16,      # ?
                 pagecat_names_embedding_size=3,     # ?

                 deep_layers_activation=tf.nn.relu,
                 cvr_deep_layers=(128, 64, 32),    # 512, 256, 128
                 ctr_deep_layers=(128, 64, 32),    # 512, 256, 128
                 cvr_dropout_deep_train=(0.90, 0.90, 0.9, 0.9),
                 ctr_dropout_deep_train=(0.90, 0.90, 0.9, 0.9),
                 cvr_dropout_deep_test=(1.0, 1.0, 1.0, 1.0),
                 ctr_dropout_deep_test=(1.0, 1.0, 1.0, 1.0),
                 learning_rate=0.0005,
                 epoch=100,
                 batch_size=20480,
                 output_dir="",
                 eval_metric=roc_auc_score
                 ):

        # 初始化离散值特征
        self.cat_names_index = None
        self.cat_names_weight = None
        self.cat1_names_index = None
        self.cat1_names_weight = None
        self.brand_names_index = None
        self.brand_names_weight = None
        self.pagecat_names_index = None
        self.pagecat_names_weight = None

        # 初始化连续值特征
        self.dense_feature_num = 14
        self.pass_avg_core = None
        self.pass_total_num = None
        self.gross_margin = None
        self.exposure_sales_volume = None
        self.exposure_sales_rate = None
        self.week2sales = None
        self.week2sales_volume = None
        self.daily_rate = None
        self.total_favorite_count = None
        self.youtube = None
        self.stock_flag = None
        self.shop_price = None
        self.display_price = None
        self.discount = None
        self.cvr_label = None
        self.ctr_label = None

        # 初始化嵌入层尺寸
        self.cat_names_feature_size = cat_names_feature_size
        self.cat_names_embedding_size = cat_names_embedding_size
        self.cat_names_dense = None

        self.cat1_names_feature_size = cat1_names_feature_size
        self.cat1_names_embedding_size = cat1_names_embedding_size
        self.cat1_names_dense = None

        self.brand_names_feature_size = brand_names_feature_size
        self.brand_names_embedding_size = brand_names_embedding_size
        self.brand_names_dense = None

        self.pagecat_names_feature_size = pagecat_names_feature_size
        self.pagecat_names_embedding_size = pagecat_names_embedding_size
        self.pagecat_names_dense = None

        # 初始化嵌入层
        self.embedding_layer = None
        self.input_layer = None

        self.cat_names_embedding_table = None
        self.cat1_names_embedding_table = None
        self.brand_names_embedding_table = None
        self.pagecat_names_embedding_table = None

        self.deep_layers_activation = deep_layers_activation
        self.cvr_deep_layers = cvr_deep_layers
        self.ctr_deep_layers = ctr_deep_layers

        # 初始化隐藏层
        self.cvr_deep_weights = dict()
        self.ctr_deep_weights = dict()
        self.cvr_dropout_deep_train = cvr_dropout_deep_train
        self.ctr_dropout_deep_train = ctr_dropout_deep_train
        self.cvr_dropout_deep_test = cvr_dropout_deep_test
        self.ctr_dropout_deep_test = ctr_dropout_deep_test
        self.cvr_dropout_deep = None
        self.ctr_dropout_deep = None
        self.cvr_deep_output = None
        self.ctr_deep_output = None

        # 初始化输出层
        self.cvr_predictions = None
        self.ctr_predictions = None
        self.purchase_predictions = None
        self.cvr_loss = None
        self.ctr_loss = None
        self.loss = None

        # 初始化训练配置
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = None
        self.train_op = None
        self.learning_rate = learning_rate
        self.sess = None
        self.saver = None
        self.output_dir = output_dir
        self.eval_metric = eval_metric

    # 初始化输入配置
    def build_place_holders(self):
        # 样本3级分类ID与权重特征
        self.cat_names_index = tf.placeholder(tf.int32, shape=[None], name="cat_names_index")
        self.cat_names_weight = tf.placeholder(tf.float32, shape=[None], name="cat_names_weight")

        # 样本2级分类ID与权重特征
        self.cat1_names_index = tf.placeholder(tf.int32, shape=[None], name="cat1_names_index")
        self.cat1_names_weight = tf.placeholder(tf.float32, shape=[None], name="cat1_names_weight")

        # 样本品牌ID与权重特征
        self.brand_names_index = tf.placeholder(tf.int32, shape=[None], name="brand_names_index")
        self.brand_names_weight = tf.placeholder(tf.float32, shape=[None], name="brand_names_weight")

        # 样本关键词ID与权重特征
        self.pagecat_names_index = tf.placeholder(tf.int32, shape=[None], name="pagecat_names_index")
        self.pagecat_names_weight = tf.placeholder(tf.float32, shape=[None], name="pagecat_names_weight")

        # 样本数值特征
        self.pass_avg_core = tf.placeholder(tf.float32, shape=[None], name="pass_avg_score")
        self.pass_total_num = tf.placeholder(tf.float32, shape=[None], name="pass_total_num")
        self.gross_margin = tf.placeholder(tf.float32, shape=[None], name="gross_margin")
        self.exposure_sales_volume = tf.placeholder(tf.float32, shape=[None], name="exposure_sales_volume")
        self.exposure_sales_rate = tf.placeholder(tf.float32, shape=[None], name="exposure_sales_rate")
        self.week2sales = tf.placeholder(tf.float32, shape=[None], name="week2sales")
        self.week2sales_volume = tf.placeholder(tf.float32, shape=[None], name="week2sales_volume")
        self.daily_rate = tf.placeholder(tf.float32, shape=[None], name="daily_rate")
        self.total_favorite_count = tf.placeholder(tf.float32, shape=[None], name="total_favorite_count")
        self.youtube = tf.placeholder(tf.float32, shape=[None], name="youtube")
        self.stock_flag = tf.placeholder(tf.float32, shape=[None], name="stock_flag")
        self.shop_price = tf.placeholder(tf.float32, shape=[None], name="shop_price")
        self.display_price = tf.placeholder(tf.float32, shape=[None], name="display_price")
        self.discount = tf.placeholder(tf.float32, shape=[None], name="discount")

        # 样本CTR、CVR正负标记
        self.cvr_label = tf.placeholder(tf.float32, shape=[None], name="cvr_label")
        self.ctr_label = tf.placeholder(tf.float32, shape=[None], name="ctr_label")

        self.cvr_dropout_deep = self.cvr_dropout_deep_test
        self.ctr_dropout_deep = self.ctr_dropout_deep_test

    # 初始化离散特征嵌入层参数
    def build_dense_weight(self):
        # 随机数初始化嵌入层
        self.cat_names_embedding_table = tf.Variable(
            tf.random_normal([self.cat_names_feature_size, self.cat_names_embedding_size], 0.0, 0.01),
            name="cat_names_embeddings")

        self.cat1_names_embedding_table = tf.Variable(
            tf.random_normal([self.cat1_names_feature_size, self.cat1_names_embedding_size], 0.0, 0.01),
            name="cat1_names_embeddings")

        self.brand_names_embedding_table = tf.Variable(
            tf.random_normal([self.brand_names_feature_size, self.brand_names_embedding_size], 0.0, 0.01),
            name="brand_names_embeddings")

        self.pagecat_names_embedding_table = tf.Variable(
            tf.random_normal([self.pagecat_names_feature_size, self.pagecat_names_embedding_size], 0.0, 0.01),
            name="pagecat_names_embeddings")

    # 初始化隐藏层参数
    def build_deep_weight(self):
        # glorot方法初始化CVR隐藏层（glorot初始化适用于多隐藏层深度网络的参数初始化，详细请查阅论文）
        num_layer = len(self.cvr_deep_layers)
        input_size = self.dense_feature_num + self.cat_names_embedding_size + self.cat1_names_embedding_size + \
            self.brand_names_embedding_size + self.pagecat_names_embedding_size

        glorot = np.sqrt(2.0 / (input_size + self.cvr_deep_layers[0]))
        self.cvr_deep_weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.cvr_deep_layers[0])), dtype=np.float32)
        self.cvr_deep_weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot,
                                                      size=(1, self.cvr_deep_layers[0])),
                                                      dtype=np.float32)
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.cvr_deep_layers[i-1] + self.cvr_deep_layers[i]))
            self.cvr_deep_weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.cvr_deep_layers[i-1], self.cvr_deep_layers[i])),
                dtype=np.float32)
            self.cvr_deep_weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.cvr_deep_layers[i])),
                dtype=np.float32)

        # glorot方法初始化CTR隐藏层
        num_layer = len(self.ctr_deep_layers)
        input_size = self.dense_feature_num + self.cat_names_embedding_size + self.cat1_names_embedding_size + \
            self.brand_names_embedding_size + self.pagecat_names_embedding_size

        glorot = np.sqrt(2.0 / (input_size + self.ctr_deep_layers[0]))
        self.ctr_deep_weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.ctr_deep_layers[0])), dtype=np.float32)
        self.ctr_deep_weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot,
                                                                       size=(1, self.ctr_deep_layers[0])),
                                                      dtype=np.float32)
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.ctr_deep_layers[i-1] + self.ctr_deep_layers[i]))
            self.ctr_deep_weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.ctr_deep_layers[i-1], self.ctr_deep_layers[i])),
                dtype=np.float32)
            self.ctr_deep_weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.ctr_deep_layers[i])),
                dtype=np.float32)

    # 构造连续特征输入层
    def build_input_layer(self):
        pass_avg_core = tf.reshape(self.pass_avg_core, shape=[-1, 1])
        pass_total_num = tf.reshape(self.pass_total_num, shape=[-1, 1])
        gross_margin = tf.reshape(self.gross_margin, shape=[-1, 1])
        exposure_sales_volume = tf.reshape(self.exposure_sales_volume, shape=[-1, 1])
        exposure_sales_rate = tf.reshape(self.exposure_sales_rate, shape=[-1, 1])
        week2sales = tf.reshape(self.week2sales, shape=[-1, 1])
        week2sales_volume = tf.reshape(self.week2sales_volume, shape=[-1, 1])
        daily_rate = tf.reshape(self.daily_rate, shape=[-1, 1])
        total_favorite_count = tf.reshape(self.total_favorite_count, shape=[-1, 1])
        youtube = tf.reshape(self.youtube, shape=[-1, 1])
        stock_flag = tf.reshape(self.stock_flag, shape=[-1, 1])
        shop_price = tf.reshape(self.shop_price, shape=[-1, 1])
        display_price = tf.reshape(self.display_price, shape=[-1, 1])
        discount = tf.reshape(self.discount, shape=[-1, 1])
        self.input_layer = tf.concat([self.embedding_layer,
                                      pass_avg_core,
                                      pass_total_num,
                                      gross_margin,
                                      exposure_sales_volume,
                                      exposure_sales_rate,
                                      week2sales,
                                      week2sales_volume,
                                      daily_rate,
                                      total_favorite_count,
                                      youtube,
                                      stock_flag,
                                      shop_price,
                                      display_price,
                                      discount], axis=-1)

    # 构造离散特征嵌入层
    def build_dense_layer(self):
        cat_names_embedding = tf.nn.embedding_lookup(self.cat_names_embedding_table, self.cat_names_index)
        cat_names_value = tf.reshape(self.cat_names_weight, shape=[-1, 1])
        self.cat_names_dense = tf.multiply(cat_names_embedding, cat_names_value)

        cat1_names_embedding = tf.nn.embedding_lookup(self.cat1_names_embedding_table, self.cat1_names_index)
        cat1_names_value = tf.reshape(self.cat1_names_weight, shape=[-1, 1])
        self.cat1_names_dense = tf.multiply(cat1_names_embedding, cat1_names_value)

        brand_names_embedding = tf.nn.embedding_lookup(self.brand_names_embedding_table, self.brand_names_index)
        brand_names_value = tf.reshape(self.brand_names_weight, shape=[-1, 1])
        self.brand_names_dense = tf.multiply(brand_names_embedding, brand_names_value)

        pagecat_names_embedding = tf.nn.embedding_lookup(self.pagecat_names_embedding_table, self.pagecat_names_index)
        pagecat_names_value = tf.reshape(self.pagecat_names_weight, shape=[-1, 1])
        self.pagecat_names_dense = tf.multiply(pagecat_names_embedding, pagecat_names_value)

        self.embedding_layer = tf.concat([self.cat_names_dense,
                                          self.cat1_names_dense,
                                          self.brand_names_dense,
                                          self.pagecat_names_dense], axis=-1)

    # 构建输出层
    def build_logit_layer(self):
        # CVR隐藏层
        cvr_deep = self.input_layer
        cvr_deep = tf.nn.dropout(cvr_deep, self.cvr_dropout_deep[0])
        for i in range(0, len(self.cvr_deep_layers)):
            cvr_deep = tf.add(tf.matmul(cvr_deep, self.cvr_deep_weights["layer_%d" % i]),
                              self.cvr_deep_weights["bias_%d" % i]) # None * layer[i] * 1

            cvr_deep = self.deep_layers_activation(cvr_deep)
            cvr_deep = tf.nn.dropout(cvr_deep, self.cvr_dropout_deep[1+i])  # dropout at each Deep layer
        self.cvr_deep_output = cvr_deep

        # CTR隐藏层
        ctr_deep = self.input_layer
        ctr_deep = tf.nn.dropout(ctr_deep, self.ctr_dropout_deep[0])
        for i in range(0, len(self.ctr_deep_layers)):
            ctr_deep = tf.add(tf.matmul(ctr_deep, self.ctr_deep_weights["layer_%d" % i]),
                              self.ctr_deep_weights["bias_%d" % i]) # None * layer[i] * 1

            ctr_deep = self.deep_layers_activation(ctr_deep)
            ctr_deep = tf.nn.dropout(ctr_deep, self.ctr_dropout_deep[1+i])  # dropout at each Deep layer
        self.ctr_deep_output = ctr_deep

        # 输出预测值，计算误差，指定误差优化方法
        ctr_logits = tf.layers.dense(self.ctr_deep_output, 1, activation=None)
        self.cvr_predictions = tf.sigmoid(tf.layers.dense(self.cvr_deep_output, 1, activation=None))
        self.ctr_predictions = tf.sigmoid(ctr_logits)
        ctr_logits = tf.reshape(ctr_logits, shape=[-1])
        self.cvr_predictions = tf.reshape(self.cvr_predictions, shape=[-1], name="CVR")
        self.ctr_predictions = tf.reshape(self.ctr_predictions, shape=[-1], name="CTR")
        self.purchase_predictions = tf.multiply(self.ctr_predictions, self.cvr_predictions, name="CTCVR")

        self.cvr_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(self.cvr_label,
                                                                           self.purchase_predictions), name="cvr_loss")
        self.ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ctr_label,
                                                                              logits=ctr_logits), name="ctr_loss")
        self.loss = tf.add(self.ctr_loss, self.cvr_loss, name="ctcvr_loss")

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    # 初始化会话
    def initialize_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    # 重新载入会话
    def restore_session(self):
        log_handler.log.info("Reloading the latest trained model...")
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.saver.restore(self.sess, self.output_dir)

    # 保存会话
    def save_session(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.saver.save(self.sess, self.output_dir)

    # 清理当前线程中的所有TF变量与会话
    @staticmethod
    def clear_all():
        tf.reset_default_graph()

    # 迭代优化主流程
    def run_epoch(self, train_data, test_data):
        current_best_loss = 1000000000
        for i in range(self.epoch):
            log_handler.log.info(16*"-" + "%s th iteration" % i + 16*"-")
            total_sample_num = len(train_data["purchased"].values.tolist())
            for j in range(0, total_sample_num, self.batch_size):
                fd = self.feed_batch(train_data, j, self.batch_size)
                train_loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=fd)
            loss, purchase_pred_score, click_pred_score = self.evaluate(test_data)
            log_handler.log.info("New training loss aquired: %s" % loss)
            log_handler.log.info("New purchase prediction score: %s" % purchase_pred_score)
            log_handler.log.info("New click prediction score: %s" % click_pred_score)

            if loss < current_best_loss:
                current_best_loss = loss
                log_handler.log.info("New best training loss aquired: %s" % current_best_loss)
                self.save_session()

    # 饲喂分批样本
    def feed_batch(self, train_data, index, batch_size):
        real_batch_size = len(train_data["cat"][index:index + batch_size].values.tolist())
        feed_dict = {
            # 样本离散特征
            self.cat_names_index: train_data["cat"][index:index + batch_size].values.tolist(),
            self.cat_names_weight: np.ones(real_batch_size, dtype=float).tolist(),
            self.cat1_names_index: train_data["cat1"][index:index + batch_size].values.tolist(),
            self.cat1_names_weight: np.ones(real_batch_size).tolist(),
            self.brand_names_index: train_data["brand"][index:index + batch_size].values.tolist(),
            self.brand_names_weight: np.ones(real_batch_size).tolist(),
            self.pagecat_names_index: train_data["pagecat"][index:index + batch_size].values.tolist(),
            self.pagecat_names_weight: np.ones(real_batch_size).tolist(),

            # 样本连续特征
            self.pass_avg_core: train_data["avg_score"][index:index + batch_size].values.tolist(),
            self.pass_total_num: train_data["score_num"][index:index + batch_size].values.tolist(),
            self.gross_margin: train_data["gross_margin"][index:index + batch_size].values.tolist(),
            self.exposure_sales_volume: train_data["exposure_sales_volume"][index:index + batch_size].values.tolist(),
            self.exposure_sales_rate: train_data["exposure_sales_rate"][index:index + batch_size].values.tolist(),
            self.week2sales: train_data["week2sales"][index:index + batch_size].values.tolist(),
            self.week2sales_volume: train_data["week2sales_volume"][index:index + batch_size].values.tolist(),
            self.daily_rate: train_data["daily_rate"][index:index + batch_size].values.tolist(),
            self.total_favorite_count: train_data["favorite"][index:index + batch_size].values.tolist(),
            self.youtube: train_data["youtube"][index:index + batch_size].values.tolist(),
            self.stock_flag: train_data["stock_flag"][index:index + batch_size].values.tolist(),
            self.shop_price: train_data["shop_price"][index:index + batch_size].values.tolist(),
            self.display_price: train_data["display_price"][index:index + batch_size].values.tolist(),
            self.discount: train_data["discount"][index:index + batch_size].values.tolist(),

            # 样本标记
            self.cvr_label: train_data["purchased"][index:index + batch_size].values.tolist(),
            self.ctr_label: train_data["clicked"][index:index + batch_size].values.tolist()
        }
        return feed_dict

    # 模型训练
    def train(self, train_data, test_data):
        self.build_place_holders()
        self.build_dense_weight()
        self.build_deep_weight()
        self.build_dense_layer()
        self.build_input_layer()
        self.build_logit_layer()
        self.initialize_session()
        self.run_epoch(train_data, test_data)

    # 模型预测
    def predict(self, test_data, result_file):
        total_sample_num = len(test_data["pagecat"].values.tolist())
        test_data["purchased"] = np.zeros(total_sample_num, dtype="float")
        test_data["clicked"] = np.zeros(total_sample_num, dtype="float")
        fd = self.feed_batch(test_data, 0, total_sample_num)
        purchase_prob, click_prob = self.sess.run([self.purchase_predictions, self.ctr_predictions], feed_dict=fd)

        test_data["click_prob"] = click_prob
        test_data["purchase_prob"] = purchase_prob
        test_data["overall_prob"] = test_data["purchase_prob"].multiply(test_data["click_prob"])
        test_data = test_data[["goodssn", "keyword", "purchase_prob", "click_prob", "overall_prob"]]
        pd.to_pickle(test_data, get_data_path()+result_file)

    # 模型评估
    def evaluate(self, test_data):
        total_sample_num = len(test_data["purchased"].values.tolist())
        fd = self.feed_batch(test_data, 0, total_sample_num)
        loss, purchase_pred, click_pred = self.sess.run([self.loss, self.purchase_predictions, self.ctr_predictions],
                                                        feed_dict=fd)
        cvr_label = np.array(test_data["purchased"].values.tolist())
        ctr_label = np.array(test_data["clicked"].values.tolist())

        purchase_pred = [1 if label >= 0.5 else 0 for label in purchase_pred]
        click_pred = [1 if label >= 0.5 else 0 for label in click_pred]
        cvr_label = [int(label) for label in cvr_label]
        ctr_label = [int(label) for label in ctr_label]

        log_handler.log.info("actually purchased: %s" % sum(cvr_label))
        log_handler.log.info("actually clicked: %s" % sum(ctr_label))

        log_handler.log.info("predict purchased: %s" % sum(purchase_pred))
        log_handler.log.info("predict clicked: %s" % sum(click_pred))

        test_data["purchased_pred"] = np.array(purchase_pred)
        test_data["clicked_pred"] = np.array(click_pred)
        test_data[["goodssn", "keyword", "expose_total", "click_total",
                   "purchase_total", "click_rate", "purchase_rate",
                   "clicked", "purchased", "clicked_pred", "purchased_pred"]].to_csv(get_data_path()+"final_result.csv")

        purchase_pred_score = self.eval_metric(cvr_label, purchase_pred)
        click_pred_score = self.eval_metric(ctr_label, click_pred)
        return loss, purchase_pred_score, click_pred_score

    # 模型测试
    def test_run(self, train_data):
        self.build_place_holders()
        self.build_dense_weight()
        self.build_deep_weight()
        self.build_dense_layer()
        self.build_input_layer()
        self.build_logit_layer()
        feed_dict = self.feed_batch(train_data, 0, 10)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            """
            for embedding in session.run([self.cat_names_embedding_table,
                                          self.cat1_names_embedding_table,
                                          self.brand_names_embedding_table,
                                          self.pagecat_names_embedding_table,
                                          ]):
                print("-"*32)
                print(embedding, embedding.shape)
            """
            for array in session.run([self.cat_names_dense,
                                      self.cat1_names_dense,
                                      self.brand_names_dense,
                                      self.pagecat_names_dense,
                                      self.embedding_layer,
                                      self.input_layer,
                                      self.cvr_deep_output,
                                      self.ctr_deep_output,
                                      self.cvr_predictions,
                                      self.ctr_predictions,
                                      self.purchase_predictions,
                                      self.cvr_loss,
                                      self.ctr_loss,
                                      self.loss
                                      ],
                                     feed_dict=feed_dict):
                print("-"*32)
                print(array, array.shape)


def esmm_model_train(train_data_path):
    # 确定嵌入层维度
    cat2id = read_data_pickle("cat2id.pkl")
    cat12id = read_data_pickle("cat12id.pkl")
    brand2id = read_data_pickle("brand2id.pkl")
    pagecat2id = read_data_pickle("pagecat2id.pkl")

    cat_feat_size = len(cat2id.keys())
    cat1_feat_size = len(cat12id.keys())
    brand_feat_size = len(brand2id.keys())
    pagecat_feat_size = len(pagecat2id.keys())

    # 初始化模型
    model = EsmmModel(
        cat_names_feature_size=cat_feat_size,
        cat_names_embedding_size=16,
        cat1_names_feature_size=cat1_feat_size,
        cat1_names_embedding_size=8,
        brand_names_feature_size=brand_feat_size,
        brand_names_embedding_size=32,
        pagecat_names_feature_size=pagecat_feat_size,
        pagecat_names_embedding_size=64,
        output_dir=get_data_path()+"../saved_sessions/result/model/"
    )

    # 训练模型，并保存
    df_train = pd.read_csv(get_data_path() + train_data_path, dtype={
        "pagecat": int,
        "brand": int,
        "cat": int,
        "cat1": int
    })
    model.train(df_train, df_train)
    model.clear_all()


def get_trained_model():
    cat2id = read_data_pickle("cat2id.pkl")
    cat12id = read_data_pickle("cat12id.pkl")
    brand2id = read_data_pickle("brand2id.pkl")
    pagecat2id = read_data_pickle("pagecat2id.pkl")

    cat_feat_size = len(cat2id.keys())
    cat1_feat_size = len(cat12id.keys())
    brand_feat_size = len(brand2id.keys())
    pagecat_feat_size = len(pagecat2id.keys())

    # 初始化模型
    model = EsmmModel(
        cat_names_feature_size=cat_feat_size,
        cat_names_embedding_size=16,
        cat1_names_feature_size=cat1_feat_size,
        cat1_names_embedding_size=8,
        brand_names_feature_size=brand_feat_size,
        brand_names_embedding_size=32,
        pagecat_names_feature_size=pagecat_feat_size,
        pagecat_names_embedding_size=64,
        output_dir=get_data_path()+"../saved_sessions/result/model/"
    )
    model.build_place_holders()
    model.build_dense_weight()
    model.build_deep_weight()
    model.build_dense_layer()
    model.build_input_layer()
    model.build_logit_layer()
    model.restore_session()

    return model


def model_train():
    try:
        log_handler.log.info("----------------Training Esmm Model----------------")
        rmdir_if_exist(get_data_path()+"../saved_sessions/result/model")
        esmm_model_train(FEATURE_TRAIN_SOURCE)
        log_handler.log.info("----------------Finish training Esmm Model----------------")
        time.sleep(5)
    except Exception:
        log_handler.log.info("----------------Error training Esmm Model----------------")
        log_handler.log.info(str(excp_trace()))
        raise Exception


if __name__ == '__main__':
    model_train()
