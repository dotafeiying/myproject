"""myproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url,include
from django.contrib import admin
from django.views.generic import TemplateView
from . import views

urlpatterns = [
    # url(r'^index/', add.index),
    url(r'^upload/$', views.upload, name='upload'),
    url(r'^get_table/$', views.get_table, name='get_table'),
    url(r'^get_table_limit/$', views.get_table_limit, name='get_table_limit'),
    url(r'^get_enb_tree/$', views.getEnbTree, name='get_enb_tree'),
    url(r'^get_choice/$', views.getChoice, name='get_choice'),
    url(r'^export_data/$', views.exportData, name='exportData'),
    url(r'^download/$', views.download, name='download'),
    url(r'^analyze/$', views.analyze, name='analyze'),
    url(r'^get_result/$', views.get_result, name='get_result'),
    url(r'^analyze_one/$', views.analyzeOne, name='analyze_one'),
    url(r'^compute_cluster/$', views.computeCluster, name='compute_cluster'),
    url(r'^job_manage/$', views.job_manage, name='job_manage'),
    url(r'^job_manage/(?P<task_id>[0-9a-zA-Z\-]+)/kill/$', views.job_kill, name='job_kill'),
    url(r'^get_job_result/$', views.get_job_result, name='get_job_result'),
    # url(r'^analyze_websocket/$', views.analyze_websocket, name='analyze_websocket'),
    url(r'^add1/$', views.add1, name='add'),

]
