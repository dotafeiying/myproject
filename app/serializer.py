#! -*- coding:utf-8 -*-
from rest_framework import serializers
from .models import Job,File
from djcelery.models import TaskMeta



class JobSerializer(serializers.ModelSerializer):
    tasks = serializers.SerializerMethodField()

    def get_tasks(self, row):
        #
        role_obj_list = row.taskmeta.to_dict()
        # ret = []
        # # 获取角色的id和名字
        # # 以字典的键值对方式显示
        # for item in role_obj_list:
        #     ret.append({"id": item.id, "result": item.result})
        return role_obj_list

    class Meta:
        model = Job
        fields = ('id','task_id', 'name', 'runtime', 'taskmeta', 'file','create_date','tasks')
        # depth = 1

# class TaskMetaSerializer(serializers.ModelSerializer):
#     # job_set = JobSerializer(many=True)
#     class Meta:
#         model = TaskMeta
#         fields = ('id','task_id', 'status', 'result', 'date_done', 'traceback','hidden','meta')
#         depth =1