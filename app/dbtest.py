# -*- coding: utf-8 -*-
import os,django,json
from datetime import datetime
os.environ.setdefault('DJANGO_SETTINGS_MODULE','myproject.settings')
django.setup()

from app.models import Job
from djcelery.models import TaskMeta,TaskState
from django.db.models import F
from django.core import serializers
from celery.result import AsyncResult


# type_list = Job.objects.all().prefetch_related('case_set')
# jobs = Job.objects.all()
# for job in jobs:
#     job.status=11111111
    # print(job)
# datas=json.dumps(list(jobs.values()))
# # datas=serializers.serialize('json',jobs)
# print(datas)
# print(list(Job.objects.values()[1:3]))
# for job in jobs:
    # job.status=1
    # print(job)

# tasks=TaskMeta.objects.all()
# for task in tasks:
#     # task.result=''
#     # task.save()
#     result=task.result
#     print(result)
    # try:
    #     res=json.loads(result)
    #     print(res)
    # except Exception as e:
    #     print(result)

# res=AsyncResult('baee076d-8caf-428a-b4d8-b78efc57a644')
# print(res.get())
# print(json.loads(res.get())['df_busy_info'])
# taskstate = Job.objects.filter(task_id='cb884b6e-209c-4073-bdcb-52dc2dddb993')
# print(taskstate.values())
# now=datetime.now()
# job=Job.objects.filter(task_id='cb884b6e-209c-4073-bdcb-52dc2dddb993').update(runtime=F('create_date'))
# print(job)

# from celery.task.control import broadcast, revoke, rate_limit,inspect
# # from celery.app.control import revoke
# actives=inspect().query_task('09ceab6c-9d4f-441d-bf0a-9bf49911e158')
# print(actives)
# for k,v in actives.items():
#     print(k)
#     print([i.get('id') for i in v])
from celery import states
ALL_STATES = sorted(states.ALL_STATES)
TASK_STATE_CHOICES = sorted(zip(ALL_STATES, ALL_STATES))
# print(TASK_STATE_CHOICES)

# jobs=Job.objects.all().prefetch_related('taskmeta')
# print(jobs.values())

from dss.Serializer import serializer
jobs=Job.objects.all()
data = serializer(jobs,datetime_format='string',foreign=True)
print(data[0])

# from app.serializer import JobSerializer
# jobs=Job.objects.all()
# serializer = JobSerializer(jobs, many=True)
# print(serializer.data)

# taskmetas=TaskMeta.objects.all()
# serializer1=TaskMetaSerializer(taskmetas,many=True)
# print(serializer1.data)

