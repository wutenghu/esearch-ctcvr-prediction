# -*- coding:utf-8 -*-
#   AUTHOR: 张耘健
#     DATE: 2019-02-20
#     DESC: 处理请求

import ast
from flask import request
import threading
from threading import Lock
from flask.views import View
from flask import jsonify
from load_result.load_result_main import load_result_main
from load_result.load_to_db import check_algorithm
from load_result.load_to_db import update_algorithm
from load_result.inform_redis import inform_redis_msg
from build_everything.build_raw_sample import build_raw_sample
from build_everything.build_goods_detail import download_goods_detail
from build_everything.build_sample import build_sample
from model.esmm_model import model_train
from traceback import format_exc as excp_trace
from log_handler.Log import log_handler

GB_CTCVR_LOCK = Lock()


class HandleUrl(View):
    def __init__(self):
        self.request = request
        self.args = request.args  # 该请求的所有参数, 返回字典
        self.form = request.form
        try:
            keyword = self.args['type']
        except Exception as e:
            keyword = None
        try:
            key = self.args['key']
        except Exception as e:
            key = '{}'
        try:
            key2 = self.form['key']
        except Exception as e:
            key2 = '{}'

        args = ast.literal_eval(key)
        form = ast.literal_eval(key2)

        if isinstance(args, list) or isinstance(form, list):
            if args:
                self.settings = args
            else:
                self.settings = form
        else:
            self.settings = self.get_args(args, form)
        self.keyword = keyword

    def get_args(self, args=None, form=None):
        params = {}
        if args:
            params.update({k: v for k, v in args.items()})

        if form:
            params.update({k: v for k, v in form.items()})
        return params


class GBCTCVRHandleUrl(HandleUrl):
    def __init__(self):
        super(GBCTCVRHandleUrl, self).__init__()

    def dispatch_request(self):
        check_result, result_msg = self.check_params(self.settings, self.keyword)
        if not check_result:
            response = {
                'code': 3,
                'text': 'Bad params for GB CTCVR!!!' + result_msg,
                'args': self.settings
            }
            return jsonify(response)
        else:
            setting = self.settings[0]
            agent = setting["agent"]
            bts_policy = setting["bts_policy"]
            domain = setting["domain"]
            channel = setting["channel"]
            algorithm_id = setting["id"]
            keyword = self.keyword
            redis_msg = "#".join([domain, agent, keyword, bts_policy, str(algorithm_id)])

        try:
            t = threading.Thread(target=gbmain, args=(agent, bts_policy, domain, channel, algorithm_id, redis_msg))
            t.start()
            response = {
                'code': 2,
                'text': 'GB CTCVR running request submitted',
                'args': self.settings
            }
        except:
            response = {
                'code': 3,
                'text': 'GB CTCVR running request submitting failed',
                'args': self.settings
            }
        return jsonify(response)

    @staticmethod
    def check_params(settings, keyword):
        if not (isinstance(keyword, str) or isinstance(keyword, unicode)):
            log_handler.log.info(type(keyword))
            return False, "keyword should be string"
        if not isinstance(settings, list):
            return False, "param should be list"
        if len(settings) == 0:
            return False, "param should not be empty"
        setting = settings[0]
        param_keys = {"agent", "bts_policy", "domain", "channel", "id"}
        param_keys_input = set(setting.keys())
        keys_intersection = param_keys.intersection(param_keys_input)
        if len(keys_intersection) != len(param_keys):
            return False, "param keys should as least include %s" % str(param_keys)

        if not isinstance(setting["id"], int):
            return False, "param id should be integer"

        setting_alter = dict()
        for key in param_keys:
            setting_alter[key] = setting[key]
        setting_alter.pop("id")
        for key, value in setting_alter.items():
            if not (isinstance(value, str) or isinstance(value, unicode)):
                return False, "param %s should be string" % key
        return True, "sucess"


def gbmain(agent, bts_policy, domain, channel, algorithm_id, redis_msg):
    if GB_CTCVR_LOCK.acquire(False):
        try:
            log_handler.log.info("**Server**: GB CTCVR running request accepted, task lock aquired")
            if check_algorithm(algorithm_id):
                log_handler.log.info("**Server**: Inform DB that GB CTCVR is running")
                update_algorithm(algorithm_id, 1, domain)
                build_raw_sample()
                download_goods_detail()
                build_sample()
                model_train()
                load_result_main(agent, bts_policy)
                log_handler.log.info("**Server**: Inform DB that GB CTCVR has finished running")
                update_algorithm(algorithm_id, 2, domain)
                log_handler.log.info("**Server**: Inform Redis that GB CTCVR has finished running, msg:%s" % redis_msg)
                inform_redis_msg(channel, redis_msg)
                log_handler.log.info("**Server**: GB CTCVR complete, mission successful")
            else:
                log_handler.log.info("**Server**: Algorithm with id:%s not found" % algorithm_id)
        except:
            error_msg = str(excp_trace())
            log_handler.log.info("**Server**: Inform DB that GB CTCVR running fatal：%s" % error_msg)
            update_algorithm(algorithm_id, 3, domain)
        finally:
            GB_CTCVR_LOCK.release()
            log_handler.log.info("**Server**: Request done processing, task lock released")
    else:
        log_handler.log.info("**Server**: GB CTCVR running request rejected, for another instance is already running!!!")
