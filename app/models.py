from django.db import models
from django.utils import timezone
from djcelery.models import TaskMeta

# Create your models here.
# from .fileStorage.storage import FileStorage
# pic=models.FileField(upload_to='upload/%Y/%m/%d',storage=FileStorage())  #如果上传文件可以将ImageField换为FileField

class File(models.Model):
    name = models.CharField(max_length = 50)
    file = models.FileField(upload_to='upload/%Y/%m/%d')
    size = models.CharField(max_length = 50)

    def __str__(self):
        return self.name

class SqlModel(models.Model):
    count_sql = models.TextField(max_length = 255,blank=False)
    query_sql = models.TextField(max_length=255,blank=False)
    export_sql = models.TextField(max_length=255,blank=False)
    create_date = models.DateTimeField('创建时间', default=timezone.now)

    def __str__(self):
        return self.count_sql

class Job(models.Model):
    # jobid = models.CharField(max_length = 255)
    task_id = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    create_date = models.DateTimeField('提交时间', default=timezone.now)
    runtime = models.FloatField('运行时间', null=True, help_text='in seconds if task succeeded')
    file = models.ForeignKey(File,verbose_name='导入文件',null=True,
                                 on_delete=models.SET_NULL)
    taskmeta=models.OneToOneField(TaskMeta,on_delete=models.CASCADE,null=True)

    class Meta:
        ordering = ['-create_date']

    def __str__(self):
        return self.task_id
