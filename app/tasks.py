from __future__ import absolute_import
from celery import shared_task,task,Task
from celery.utils.log import get_task_logger
from celery.signals import after_task_publish,task_prerun,task_postrun,task_revoked
import time,json
from datetime import datetime
import redis
from core.redis_helper import RedisHelper
from core.conf import config

from djcelery.models import TaskMeta
from django.db.models import F
from .models import Job

# print(__name__)
logger = get_task_logger(__name__)

from celery.signals import worker_process_init
@worker_process_init.connect
def fix_multiprocessing(**_):
  from multiprocessing import current_process
  try:
    current_process()._config
  except AttributeError:
    current_process()._config = {'semprefix': '/mp'}


@shared_task(track_started=True)
def add(x, y):
    print('task',x)
    time.sleep(6)
    return x + y

@task_prerun.connect
def pre_task_run(task_id, task, sender, *args, **kwargs):
    # print('pre_task_run: task_id: {body}; task: {sender}'.format(body=task_id, sender=task))
    logger.info('task [{task_id}] 开始执行, taskname: {task.name}'.format(task_id=task_id, task=task))



# @after_task_publish.connect
# def task_send_handler(sender=None, body=None, **kwargs):
#     print('after_task_publish: task_id: {body[id]}; sender: {sender}'.format(body=body, sender=sender))
#     logger.info('after_task_publish: task_id: {body[id]}; sender: {sender}'.format(body=body, sender=sender))


# @task_postrun.connect
# def post_task_run(task, task_id,state,retval,sender,signal, args, kwargs):
#     print(task_id)
#     logger.info('post_task_run: {0}, state: {1}'.format(task_id,task))

# @task_prerun.connect
# def post_task_run(*args, **kwargs):
#     print(args)
#     print(kwargs)

@task_revoked.connect
def task_revoked(request,terminated,sender,expired,signal,signum):
    now=datetime.now()
    task_id=request.id
    logger.warn('task [{0}] 被停止。'.format(task_id))
    job = Job.objects.filter(task_id=task_id).first()
    if job:
        job.runtime = (now - job.create_date).seconds
        job.save()
    # logger.info('args:', str(args))
    # logger.info('kwargs:', str(kwargs))
    # print('task_revoked: {0.id}, signum: {1}'.format(request,signum))
    # logger.info('task_revoked: {0.id}, signum: {1}'.format(request,signum))

class MyTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        # print('task done: {0}'.format(retval))
        # now=datetime.now()
        # Job.objects.filter(task_id=task_id).update(runtime=(now-F('create_date')).seconds)
        job=Job.objects.filter(task_id=task_id).first()
        if job:
            channel = job.id
            print('channel:', channel)
            redis_helper = RedisHelper(channel)
            redis_helper.public('task [{0}] success。'.format(task_id))
        logger.info('task [{0}] 执行成功, success'.format(task_id))
        return super(MyTask, self).on_success(retval, task_id, args, kwargs)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        # print('task fail, reason: {0}'.format(exc))
        # now = datetime.now()
        # jobs = Job.objects.filter(task_id=task_id)
        # if jobs.exists():
        #     job = jobs[0]
        #     job.runtime = (now - job.create_date).seconds
        #     job.save()
        job = Job.objects.filter(task_id=task_id).first()
        if job:
            channel = job.id
            print('channel:', channel)
            redis_helper = RedisHelper(channel)
            redis_helper.public('failed')
        logger.error('task [{0}] 执行失败, reason: {1} ,einfo: {2}'.format(task_id,exc,einfo))
        return super(MyTask, self).on_failure(exc, task_id, args, kwargs, einfo)

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        now = datetime.now()
        job = Job.objects.filter(task_id=task_id).first()
        if job:
            job.runtime = (now - job.create_date).seconds
            job.save()
        # print(einfo)
        # logger.info(einfo)
        # print('Task returned: {0!r}'.format(self.request))
        # logger.info('task [{0}] done: finished. status: {1}'.format(task_id,status))

# def my_monitor(app):
#     state = app.events.State()
#
#     def announce_failed_tasks(event):
#         state.event(event)
#         # task name is sent only with -received event, and state
#         # will keep track of this for us.
#         task = state.tasks.get(event['uuid'])
#
#         print('TASK FAILED: %s[%s] %s' % (
#             task.name, task.uuid, task.info(), ))
#
#     with app.connection() as connection:
#         recv = app.events.Receiver(connection, handlers={
#                 'task-failed': announce_failed_tasks,
#                 '*': state.event,
#         })
#         recv.capture(limit=None, timeout=None, wakeup=True)

@shared_task(track_started=True,bind=True,base=MyTask)
def cluster_analyze_task(self,channel,file_path,file_path_id):
    # logger.info(('Executing task id {0.id}, args: {0.args!r}'
    #              'kwargs: {0.kwargs!r}').format(self.request))
    from core.DataAnalysis import Busycell_calc
    task_id=self.request.id
    taskmeta = TaskMeta.objects.filter(task_id=task_id).first()
    Job.objects.filter(task_id=task_id).update(taskmeta=taskmeta)
    busycell = Busycell_calc(channel,file_path,task_id,file_path_id)
    df_busy_info, download_url = busycell.run()
    df_busy_info=json.loads(df_busy_info.to_json(orient='records', force_ascii=False))
    result=json.dumps({'download_url':download_url,'df_busy_info':df_busy_info})
    # return json.dumps(download_url)
    r = redis.Redis(host=config.host2, port=config.port2, db=config.db6)
    r.hset(task_id, "result", result)
    return download_url