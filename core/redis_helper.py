# -*- coding: UTF-8 -*-
import redis
import logging
from core.conf import config

logger = logging.getLogger('data_analysis')

class Logger_Redis(object):
    """
    同时在文件里和redis记录日志
    redis 里记录日志用于实时反馈结果
    """

    def __init__(self, key, logger):
        self.logger = logger
        # 本地缓存数据库 redis
        self._r = redis.Redis(host=config.host2, port=config.port2, db=config.db2)
        self.key = key

    def info(self, msg):
        self.logger.info(msg)
        self._r.lpush(self.key, msg)

    def error(self, msg):
        self.logger.error(msg)
        self._r.lpush(self.key, msg)



class RedisHelper:
    def __init__(self, key):
        self.__conn = redis.Redis(host=config.host2, port=config.port2, db=config.db2)
        self.chan_sub = key
        self.chan_pub= key

    #发送消息
    def public(self,msg):
        # logger.info(msg)
        self.__conn.publish(self.chan_pub,msg)
        return True
    #订阅
    def subscribe(self):
        #打开收音机
        pub = self.__conn.pubsub()
        #调频道
        pub.subscribe(self.chan_sub)
        #准备接收
        pub.parse_response()
        return pub