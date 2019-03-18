# -*- coding:utf-8 -*-
import sys
import time
# import threading
import schedule
from server.logic import gbmain
from flask import Flask
from server.logic import GBCTCVRHandleUrl
from foundation.easy_daemon import EasyDaemon

app = Flask(__name__)

app.add_url_rule(rule='/gearbest_ctcvr', view_func=GBCTCVRHandleUrl.as_view('gb'))


def run_server():
    app.run(
        host='0.0.0.0',
        port=50001,
        threaded=True,
        # debug=DEBUG,
    )


def run_background():
    domain = "GB"
    agent = "pc"
    keyword = "keyword"
    bts_policy = "C"
    algorithm_id = 42
    channel = "rerank.db.sync.v3"
    redis_msg = "#".join([domain, agent, keyword, bts_policy, str(algorithm_id)])
    schedule.every(1).minutes.do(gbmain, agent, bts_policy, domain, channel, algorithm_id, redis_msg)
    while True:
        schedule.run_pending()
        time.sleep(1)


class MyEasyDemaen(EasyDaemon):
    def run(self):
        """
        # 定时任务+服务
        t_server = threading.Thread(target=run_server)
        t_background = threading.Thread(target=run_background)
        t_server.start()
        t_background.start()
        """

        # 服务
        app.run(
            host='0.0.0.0',
            port=50001,
            threaded=True,
            # debug=DEBUG,
        )


def main():
    daemon_conf = {
        "stdin": '/dev/null',
        "stdout": '/dev/null',
        "stderr": '/tmp/ctcvr-prediction.log',
        "pidfile": '/tmp/ctcvr-prediction.pid',
        "name": 'esearch-ctcvr-prediction'
    }
    reminder = "Error: please inter one of these following arguments: start, stop, restart"
    args = list(sys.argv)
    if len(args) < 2:
        print(reminder)
        return
    cmd = args[1]
    my_easy_demaen_obj = MyEasyDemaen(daemon_conf)
    if cmd == "start":
        my_easy_demaen_obj.start()
    elif cmd == "restart":
        my_easy_demaen_obj.restart()
    elif cmd == "stop":
        my_easy_demaen_obj.stop()
    else:
        print(reminder)


if __name__ == '__main__':
    main()
