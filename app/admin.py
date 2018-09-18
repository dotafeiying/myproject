from django.contrib import admin
from django.db import models
from djangocodemirror.fields import CodeMirrorWidget

# Register your models here.
from .models import File,SqlModel,Job
from djcelery.models import TaskMeta
# Register your models here.

class FileAdmin(admin.ModelAdmin):
    list_display = ['id', 'name','file', 'size']
    list_display_links = ('name',)

class SqlModelAdmin(admin.ModelAdmin):
    formfield_overrides = {
        models.TextField: {'widget': CodeMirrorWidget(config_name='default', embed_config=True,attrs={'cols': '100', 'rows': '20'})},
    }
    list_display = ['id', 'count_sql', 'query_sql', 'export_sql', 'create_date']
    list_display_links = ('id',)

# class JobInline(admin.TabularInline): # admin.TabularInline admin.StackedInline
#     model = Job
#     extra = 0

# class FileAdmin(admin.ModelAdmin):
#     # fieldsets = [
#     #     (None,               {'fields': ['question_text']}),
#     #     ('Date information', {'fields': ['pub_date'], 'classes': ['collapse']}),
#     # ]
#     inlines = [JobInline]

class JobAdmin(admin.ModelAdmin):
    def show_status(self, obj):
        print('m',obj.taskmeta)
        return getattr(obj.taskmeta,'status',None)
    def show_result(self, obj):
        return getattr(obj.taskmeta,'result',None)
    list_display = ['id', 'task_id','name', 'create_date','show_status','show_result']
    list_display_links = ('task_id',)

class TaskMetaAdmin(admin.ModelAdmin):
    list_display = ['id', 'task_id','status', 'result','traceback','meta']
    list_display_links = ('task_id',)

admin.site.register(File,FileAdmin)
admin.site.register(SqlModel,SqlModelAdmin)
admin.site.register(Job,JobAdmin)
admin.site.register(TaskMeta,TaskMetaAdmin)