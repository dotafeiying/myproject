# -*- coding: utf-8 -*-
import os,json,time,uuid
from io import BytesIO
from collections import defaultdict
from decimal import Decimal
from datetime import datetime,date

import xlrd
import redis
import MySQLdb
from MySQLdb.cursors import DictCursor
from impala.dbapi import connect
import pandas as pd
import numpy as np
import paramiko
from retrying import retry
from dwebsocket import accept_websocket

from django.shortcuts import render,reverse
from django.http import JsonResponse,HttpResponse,StreamingHttpResponse,FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.core import serializers
from .models import File,SqlModel,Job
from django.conf import settings
from django.db import connection

from dss.Serializer import serializer

from celery import current_app
# from celery.task.control import revoke
from celery.task.control import broadcast, revoke, rate_limit,inspect
from celery.result import AsyncResult
from djcelery.models import TaskState,TaskMeta

from core.DataAnalysis import Busycell_calc
from core.analyzeOne import analyze_one
from core.redis_helper import RedisHelper
from core import FileHandle
from .tasks import add,cluster_analyze_task
from .utils.query_sql import generate_sql
from .utils.excel2html import Excel2Html
from core.conf import config

# Create your views here.

def parse(x):
    if x is None or x == '' or pd.isnull(x):
        return True
    else:
        try:
            if isinstance(float(x),float):
                return False
            else:
                return True
        except:
            return True

def parse_fun(x):
    if x is None or x == '':
        return '%s (该值不能为空！)'%x
    else:
        try:
            if isinstance(float(x),float):
                return x
            else:
                return '%s (不合法数据类型！)'%x
        except:
            return '%s (不合法数据类型！)'%x

def filter_fun(x):
    enbid,cellid=x.enbid,x.cellid
    if enbid is None or enbid == '' or cellid is None or cellid == '':
        return False
    else:
        try:
            return isinstance(float(enbid), (float)) and isinstance(float(cellid), (float))
        except:
            return False

@csrf_exempt
def upload(request):
    ret={}
    ret['row_error'] = None
    if request.method=='POST':
        print(request.POST.get('extension'))
        print(request.FILES.get('file'))
        file_obj = request.FILES.get('file')
        name=file_obj.name
        size=file_obj.size
        ret['name'] = file_obj.name
        ret['size'] = file_obj.size

        # wb = xlrd.open_workbook(filename=file_obj.name, file_contents=next(file_obj.chunks()))  # 关键点在于这里
        wb = xlrd.open_workbook(filename=file_obj.name, file_contents=file_obj.read())  # 关键点在于这里
        table = wb.sheets()[0]
        headers=table.row_values(0)
        print(headers)
        subheaders = ['province', 'city', 'enbid', 'cellid','cellname']
        essential_headers = ['enbid','cellid']
        checkable=False
        if set(subheaders).issubset(headers):
            col_num = table.ncols
            datas = {}
            for i in range(col_num):
                datas[headers[i]]=table.col_values(i,start_rowx=1)
            df=pd.DataFrame(datas,columns=headers)
            # df = pd.DataFrame({
            #     'province': table.col_values(0, start_rowx=1),
            #     'city': table.col_values(1, start_rowx=1),
            #     'enbid': table.col_values(2, start_rowx=1),
            #     'cellid': table.col_values(3, start_rowx=1),
            #     'cellname': table.col_values(4, start_rowx=1)
            # },columns=['province','city','enbid','cellid','cellname'])
            df_bool=df.applymap(parse)
            not_essential_headers=list(set(headers).difference(set(essential_headers))) # headers中有而essential_headers中没有的
            df_bool[not_essential_headers] = False
            checkable=not any(df_bool.any().tolist())
            if checkable:
                profile = File()
                profile.name = name
                profile.size = size
                profile.file = file_obj
                profile.save()
                print('path:',profile.file.path)

                file_path=profile.file.path
                ret['status_code'] = 0
                ret['msg'] = '文件预览'
                ret['file_path'] = file_path
                ret['file_path_id'] = profile.id
            else:
                df['enbid']=df['enbid'].map(parse_fun)
                df['cellid'] = df['cellid'].map(parse_fun)
                # rowIndex_error=np.where(df[['enbid','cellid']].isnull())[0].tolist()
                # df_bool[['province', 'city', 'cellname']] = False
                rowIndex_error = np.where(df_bool)[0].tolist()
                row_error = [i + 1 for i in rowIndex_error]
                e=np.where(df_bool)
                cell_error=['R%sC%s'%(x[0],x[1]) for x in zip(e[0] + 1, e[1] + 1)]
                ret['status_code'] = 1
                ret['msg'] = '文件预览：共发现' + str(len(row_error)) + '处错误！'
                ret['row_error'] = row_error
                ret['cell_error'] = cell_error
            html = df.to_html(index=None, classes='previewtable')
            ret['result'] = html
        else:
            ret['result'] = '错误原因：上传文件必须至少包含 “province,city,enbid,cellid,cellname” 5列，具体格式请查看模板！'
            ret['status_code'] = 1
            ret['msg'] = '错误提示'
        print(ret)
        return JsonResponse(ret, safe=False)

@csrf_exempt
@accept_websocket
def get_table(request):
    if not request.is_websocket():
        POST = json.loads(request.body.decode('utf8'))
        limit = POST.get('limit')
        offset = POST.get('offset')
        name = POST.get('name', '')
        dateStart = POST.get('dateStart')
        dateEnd = POST.get('dateEnd')
        enbList = POST.get('enbList')
        indoor = POST.get('indoor', '')
        scene = POST.get('scene', '')
        freqID = POST.get('freqID', '')
        db = MySQLdb.connect("10.39.211.198", "root", "password", "busycell", charset='utf8')
        pk = generate_sql(name, dateStart, dateEnd, enbList, limit, offset, indoor=indoor, scene=scene, freqID=freqID)
        cursor = db.cursor(DictCursor)
        count_str = SqlModel.objects.get(pk=pk).count_sql
        cursor.execute(count_str)
        total = cursor.fetchall()[0].get('total', 0)
        query_str = SqlModel.objects.get(pk=pk).export_sql
        print('query_str:', query_str)
        cursor.execute(query_str)
        descs = cursor.description
        headers = [desc[0] for desc in descs]
        rows = cursor.fetchall()  # 返回结果行游标直读向前，读取一条
        # rows = [dict(zip(headers, row)) for row in rows]
        db.close()
        ret = {}
        ret['pk'] = pk
        ret['total'] = total
        ret['rows'] = rows
        ret['cols'] = [{'name': col, 'label': col} for col in headers]
        return JsonResponse(ret)
    else:
        for message in request.websocket:
            print('message:',message)
            message = None if message == None else message.decode('utf-8')
            para=json.loads(message) if message else None
            if para:
                start=datetime.now()
                db = MySQLdb.connect("10.39.211.198", "root", "password", "busycell", charset='utf8',
                                       cursorclass=MySQLdb.cursors.SSCursor)
                pk = generate_sql(**para)
                # pk = generate_sql(name, dateStart, dateEnd, enbList, limit, offset, indoor=indoor, scene=scene,freqID=freqID)
                cursor = db.cursor(DictCursor)
                count_str = SqlModel.objects.get(pk=pk).count_sql
                cursor.execute(count_str)
                total = cursor.fetchall()[0].get('total', 0)
                print(total)
                query_str = SqlModel.objects.get(pk=pk).export_sql
                # print('query_str:', query_str)
                cursor.execute(query_str)
                descs = cursor.description
                headers = [desc[0] for desc in descs]
                ret = {}
                ret['num'] = 0
                ret['code'] = 0
                ret['total'] = total
                i=0
                batch=total if total<=10000 else 1000 if int(total/10)<=10000 else 10000
                while True:
                    i+=1
                    row = cursor.fetchmany(batch)
                    if not row:
                        end = datetime.now()
                        duration=(end-start).seconds
                        ret['code']=1
                        ret['duration'] = duration
                        ret['pk'] = pk
                        ret['cols'] = [{'name': col, 'label': col} for col in headers]
                        msg = json.dumps(ret, cls=ComplexEncoder)
                        request.websocket.send(msg.encode('utf-8'))
                        break
                    ret['num'] = ret['num'] + len(row)
                    print(ret['num'])
                    ret['rows'] = row
                    msg = json.dumps(ret, cls=ComplexEncoder)
                    request.websocket.send(msg.encode('utf-8'))
                    time.sleep(0.1)
                cursor.close()
                db.close()
            else:
                request.websocket.close()



def getEnbTree(request):
    tabSelected=request.GET.get('tabSelected')
    db = MySQLdb.connect("10.39.211.198", "root", "password", "busycell", charset='utf8')
    res = {}
    if tabSelected=='tree':
        city=request.GET.get('city','')
        # print(city)
        if city:
            # cursor = db.cursor()
            cursor = db.cursor(DictCursor)
            query_str = "select distinct city,enbid,cellid,cellname from btsinfo where city='%s' ORDER BY enbid,cellid"%city
            # print(query_str)
            cursor.execute(query_str)
            datas = cursor.fetchall()
            # print(datas)
        else:
            cursor = db.cursor()
            query_str = "select distinct city from btsinfo"
            cursor.execute(query_str)
            datas = cursor.fetchall()
            datas= [data[0] for data in datas]
            # print(datas)
    elif tabSelected=='search':
        cursor = db.cursor(DictCursor)
        query_str = "select distinct enbid,cellid,cellname from btsinfo ORDER BY enbid,cellid"
        cursor.execute(query_str)
        datas = cursor.fetchall()
        # datas = [data[0] for data in datas]
        # print(datas)
    else:
        datas=None
    res['data']=datas
    print(datas)
    return JsonResponse(res)

def getChoice(request):
    choices=['scene','freqID']
    db = MySQLdb.connect("10.39.211.198", "root", "password", "busycell", charset='utf8')
    cursor = db.cursor()
    sql="select distinct {0} from btsinfo where {1}".format((','.join(choices)),'and '.join([item+' is not null ' for item in choices]))
    # print(sql)
    cursor.execute(sql)
    datas = cursor.fetchall()
    # print(datas)
    sceneChoices = [data[0] for data in datas]
    freqIDChoices = [data[1] for data in datas]
    # cursor.execute("select distinct scene from busycell")
    # datas = cursor.fetchall()
    # sceneChoices = [data[0] for data in datas]
    res={}
    res['sceneChoices'] = ['全部'] + list(set(sceneChoices))
    res['freqIDChoices'] = ['全部'] + list(set(freqIDChoices))
    return JsonResponse(res)

@csrf_exempt
def get_table_limit(request):
    POST = json.loads(request.body.decode('utf8'))
    limit = POST.get('limit')
    offset = POST.get('offset')
    name = POST.get('name', '')
    dateStart = POST.get('dateStart')
    dateEnd = POST.get('dateEnd')
    enbList = POST.get('enbList')
    indoor = POST.get('indoor','')
    scene = POST.get('scene', '')
    freqID = POST.get('freqID', '')
    print('indoor:',indoor)

    # cells=result.get('cell',' "" ')
    # limit=request.POST.get('limit')
    # offset=request.POST.get('offset')
    # name=request.POST.get('name','')
    # dateStart=request.POST.get('dateStart')
    # dateEnd=request.POST.get('dateEnd')
    # dataTree =request.POST.get('dataTree')

    # db = connect(host='133.21.254.164', port=21050, database='hub_yuan')
    db = MySQLdb.connect("10.39.211.198", "root", "password", "busycell", charset='utf8')
    if db.__module__ == 'impala.hiveserver2':
        cursor = db.cursor()
        count_str = "select count(*) as total from lte_busy_cell_history where (enbid like '%{0}%' or cellname like '%{0}%') " \
                    "and unix_timestamp(`finish_time`)>unix_timestamp('{1}') and unix_timestamp(`finish_time`)<unix_timestamp('{2}')".format(
            name, dateStart, dateEnd)
        print(count_str)
        cursor.execute(count_str)
        total = cursor.fetchall()[0][0]
        print(total)
        query_str = "select index,enbid,cellid,cellname,freqID,lng,lat,scene,indoor,n_cell,result,finish_time from lte_busy_cell_history where (enbid like '%{0}%' or cellname like '%{0}%') " \
                    "and unix_timestamp(`finish_time`)>unix_timestamp('{3}') and unix_timestamp(`finish_time`)<unix_timestamp('{4}') " \
                    "order by finish_time limit {2} offset {1}".format(
            name, offset, limit, dateStart, dateEnd)
        cursor.execute(query_str)
        descs = cursor.description
        headers = []
        for desc in descs:
            headers.append(desc[0])
        rows = cursor.fetchall()  # 返回结果行游标直读向前，读取一条

        # save_str = "select * from hub_yuan.lte_busy_cell_history where (enbid like '%{0}%' or cellname like '%{0}%') " \
        #            "and unix_timestamp(finish_time)>unix_timestamp('{1}') and unix_timestamp(finish_time)<unix_timestamp('{2}')".format(
        #     name, dateStart, dateEnd)
        # df=pd.read_sql(save_str,db)
        # df.to_excel('yuan.xls',index=None)
        # hostname = '133.21.254.164'
        # username = 'root'
        # password = 'qwe@321'
        # port = 22
        # ssh = paramiko.SSHClient()
        # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # ssh.connect(hostname=hostname,port=port,username=username,password=password)
        # command='''impala-shell -q "{0}" -B --output_delimiter="," --print_header -o /home/out.csv'''.format(save_str)
        # print(save_str)
        # print(command)

        # stdin, stdout, stderr = ssh.exec_command(command)
        # cursor.execute(save_str)
        # rows_save = cursor.fetchall()
        # data = tablib.Dataset(rows_save)
        # with open('test.xls', 'wb') as f:  # exl是二进制数据
        #     f.write(data.xls)
        rows = [dict(zip(headers, row)) for row in rows]
        db.close()
        ret = {}
        ret['total'] = total
        ret['rows'] = rows
        ret['cols'] = [{'name': col, 'label': col} for col in headers]
        # print(ret)
        return JsonResponse(ret)
    elif db.__module__ == 'MySQLdb.connections':
        pk=generate_sql(name, dateStart, dateEnd, enbList, limit, offset, indoor=indoor, scene=scene, freqID=freqID)
        cursor = db.cursor(DictCursor)
        count_str=SqlModel.objects.get(pk=pk).count_sql
        cursor.execute(count_str)
        total = cursor.fetchall()[0].get('total',0)
        query_str=SqlModel.objects.get(pk=pk).query_sql
        print('query_str:', query_str)
        cursor.execute(query_str)
        descs = cursor.description
        headers = [desc[0] for desc in descs]
        # for desc in descs:
        #     headers.append(desc[0])
        rows = cursor.fetchall()  # 返回结果行游标直读向前，读取一条
        # rows = [dict(zip(headers, row)) for row in rows]
        db.close()
        ret = {}
        ret['pk'] = pk
        ret['total'] = total
        ret['rows'] = rows
        ret['cols'] = [{'name': col, 'label': col} for col in headers]
        # print(ret)
        return JsonResponse(ret)




@csrf_exempt
def exportData(request):
    format = request.GET.get('format')
    pk = request.GET.get('pk')
    print('pk:',pk)
    export_sql=SqlModel.objects.get(pk=pk).export_sql
    # db = MySQLdb.connect("10.39.211.198", "root", "password", "busycell", charset='utf8')
    # cursor = db.cursor(DictCursor)
    # cursor.execute(export_sql)
    # datas = cursor.fetchall()
    if format=='csv':
        print(123)
        response = StreamingHttpResponse((row for row in FileHandle.csv_stream_response_generator(export_sql)),content_type="text/csv;charset=utf-8")
        # response = StreamingHttpResponse(stream_response_generator('csv'), content_type="text/csv")
        response['Content-Disposition'] = 'attachment; filename="query_result.csv"'
        return response
    elif format=='xlsx':
        response = StreamingHttpResponse((row for row in FileHandle.excel_stream_response_generator()), content_type="application/vnd.ms-excel")
        response['Content-Disposition'] = 'attachment; filename="result.xlsx"'
        return response

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        else:
            return json.JSONEncoder.default(self, obj)

def download(request):
    def file_iterator(file_name, chunk_size=512):
        with open(file_name,mode='rb') as f:
            while True:
                c = f.read(chunk_size)
                if c:
                    yield c
                else:
                    break
    task_id = request.GET.get('task_id')
    format = request.GET.get('format')
    if task_id:
        the_file_name=AsyncResult(task_id).result+'.'+format
    else:
        download_url = request.GET.get('filename')
        the_file_name=download_url+'.'+format
    print(the_file_name)
    response = StreamingHttpResponse(file_iterator(the_file_name))
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format('result.'+format)
    return response

@accept_websocket
def analyze(request):
    if not request.is_websocket():  # 判断是不是websocket连接
        file_path=request.GET.get('file_path')
        file_path_id = request.GET.get('file_path_id')
        print('file_path:',file_path)
        # result=analyzeTask.delay(file_path)
        # df_busy_info=result.id
        # print(df_busy_info)

        # jobid=uuid.uuid1()
        file=File.objects.get_or_create(id=file_path_id)[0]
        jobname=file.name
        job = Job(name=jobname,file=file)
        job.save()
        channel = job.id
        result = cluster_analyze_task.delay(channel,file_path,file_path_id)
        taskid=result.id
        print('taskid:',taskid)

        job.task_id=taskid
        job.save()

        return JsonResponse({'taskid': taskid,'channel': channel},safe=False)

        # busycell = Busycell_calc(file_path)
        # df_busy_info, download_url=busycell.run()
        # df_busy_info = df_busy_info.where(df_busy_info.notnull(), None)
        # rows = df_busy_info.to_dict('records')
        # for r in rows:
        #     if isinstance(r['lng'], float) and isinstance(r['lng'], float):
        #         r['lng'] = float(Decimal(r['lng']).quantize(Decimal('0.000000')))
        #         r['lat'] = float(Decimal(r['lat']).quantize(Decimal('0.000000')))
        #
        # # rows = df_busy_info.values.tolist()
        # # headers = df_busy_info.columns.tolist()
        # # print(rows)
        # # rows = [dict(zip(headers, row)) for row in rows]
        #
        # # print(df_busy_info)
        # return JsonResponse({'data':rows,'download_url':download_url}, safe=False)
    else:
        for message in request.websocket:
            print('message:',message)

            # filtpath=os.path.abspath('.')+'/log/data_analysis.log'
            # print(filtpath)
            # with open(filtpath,encoding='utf-8') as f:
            #     while True:
            #         time.sleep(1)
            #         line=f.readline()
            #         msg = line.strip()
            #         print('msg:', msg)
            #         if msg:
            #             if 'end' in msg:
            #                 print('结束')
            #                 break
            #             request.websocket.send(msg.encode('utf-8'))  # 发送消息到客户端
            #     request.websocket.send(msg.encode('utf-8'))  # 发送消息到客户端
            #     f.close()

            obj = RedisHelper(message)
            redis_sub = obj.subscribe()

            while True:
                msg = redis_sub.parse_response()
                # print('接收：', msg)
                # print('接收：', [x.decode('utf-8') for x in msg])
                msg = msg[2]
                request.websocket.send(msg)  # 发送消息到客户端
                if msg.decode()=='end':
                    break
                # if not msg:
                #     break

def get_result1(request):
    taskid=request.GET.get('taskid')
    print('taskid1:',taskid)
    res=AsyncResult(taskid).get()
    data=json.loads(res)['df_busy_info']
    # r=redis.Redis(host=config.host2, port=config.port2, db=config.db6)
    # key='celery-task-meta-' + taskid
    # print(key)
    # while True:
    #     task_result=r.get(key)
    #     if task_result:
    #         result=json.loads(task_result.decode('utf-8')).get('result')
    #         # print(result)
    #         data = json.loads(result)['df_busy_info']
    #         break
    #     time.sleep(0.2)
    return JsonResponse({'data': data},safe=False)

def get_result(request):
    taskid=request.GET.get('taskid')
    print('taskid1:',taskid)
    # res=AsyncResult(taskid).get()
    # data=json.loads(res)['df_busy_info']
    r=redis.Redis(host=config.host2, port=config.port2, db=config.db6, decode_responses=True)
    # key='celery-task-meta-' + taskid
    key=taskid
    print(key)
    while True:
        task_result=r.hget(key,'result')
        if task_result:
            data = json.loads(task_result)['df_busy_info']
            download_url=json.loads(task_result)['download_url']
            break
        else:
            time.sleep(0.2)
    return JsonResponse({'data': data,'download_url': download_url},safe=False)

@accept_websocket
def analyzeOne(request):
    if not request.is_websocket():  # 判断是不是websocket连接
        enbid=request.GET.get('enbid')
        cellid = request.GET.get('cellid')
        taskid = request.GET.get('taskid')
        eps = request.GET.get('eps')
        min_samples = request.GET.get('min_samples')
        K = request.GET.get('K')
        eps,min_samples,K=int(eps),int(min_samples),int(K)
        print('taskid:',taskid)
        print('K:',K)
        # taskID=uuid.uuid1()
        result=analyze_one(taskid,enbid,cellid,radius=eps,min_samples=min_samples,K=K)
        print(result)
        result['taskID']=taskid

        # df_busy_info=busycell.df_info
        # print(df_busy_info)
        # df_busy_info={'scheme':'yuan姐','你好':324}
        return JsonResponse(result,safe=False)
        # return HttpResponse({'enbid':enbid,'cellid':'niao'}, content_type='application/json')
        # return HttpResponse(json.dumps({'scheme':12,'result':324}), content_type='application/json')
    else:
        for message in request.websocket:
            # message=request.websocket.wait()
            print(message)

            obj = RedisHelper(message)
            redis_sub = obj.subscribe()
            while True:
                msg = redis_sub.parse_response()
                # print('接收：', msg)
                # print('接收：', [x.decode('utf-8') for x in msg])
                msg = msg[2]
                request.websocket.send(msg)  # 发送消息到客户端
                if msg.decode()=='end':
                    break

@csrf_exempt
def computeCluster(request):
    from sklearn.cluster import KMeans
    POST = json.loads(request.body.decode('utf8'))
    slider = POST.get('slider')
    # data = POST.get('slider')
    # print(slider)
    res={}
    centres_cluster = []
    for key,value in slider.items():
        data=value.get('data')
        K = value.get('K')
        print(K)
        df=pd.DataFrame(data)
        # print(df)
        X=df[['BDlng','BDlat']]
        model = KMeans(n_clusters=K, random_state=0)
        model.fit(X)
        centroid = model.cluster_centers_
        print(centroid)
        sample = model.labels_
        num_sample = np.array([len(X[sample == i]) for i in range(K)], dtype=np.int)
        # print(num_sample)
        index_cluster = np.ones(K, dtype=np.int) * int(key)
        centre_point = np.c_[centroid, num_sample, index_cluster]
        df1=pd.DataFrame(centre_point)
        print(json.loads(df1.to_json(orient='values', force_ascii=False, double_precision=6)))

        # print(centre_point)
        centres_cluster.extend(centre_point.tolist())
    print(len(centres_cluster),centres_cluster)
    res['cluster']=centres_cluster
    # print(data)
    return JsonResponse(res,safe=False)

def job_manage1(request):
    now=datetime.now()
    db = MySQLdb.connect("10.39.211.198", "root", "password", "busycell", charset='utf8')
    cursor=db.cursor(DictCursor)
    # job_query_sql = """
    #           select a.task_id,a.name,a.create_date,b.state,b.runtime
    #           from app_job a inner join djcelery_taskstate b on a.task_id=b.task_id order by a.create_date desc"""
    job_query_sql="""select task_id,name,create_date,runtime
              from app_job order by create_date desc"""
    cursor.execute(job_query_sql)
    rows = cursor.fetchall()
    cursor.close()
    db.close()
    r = redis.Redis(host=config.host2, port=config.port2, db=config.db6,decode_responses=True)
    for row in rows:
        row['state']=AsyncResult(row['task_id']).state
        row['canKill']=True if row['state']=='STARTED' else False
        row['killUrl']=reverse('job_kill', kwargs={'task_id': row['task_id']}) if row['state']=='STARTED' else None
        row['durationFormatted']=(now-row['create_date']).seconds if row['state']=='STARTED' else row['runtime']
        row['stage1PercentComplete'] = r.hget(row['task_id'], 'stage1PercentComplete') if row['state']=='STARTED' else 100
        row['stage2PercentComplete'] = r.hget(row['task_id'], 'stage2PercentComplete') if row['state'] == 'STARTED' else 100
        # if row['state'] == 'STARTED':
        #     row['stage1PercentComplete']=r.hget(row['task_id'],'stage1PercentComplete')
        #     print(r.hget(row['task_id'],'stage1PercentComplete'))
        #     row['stage2PercentComplete']=r.hget(row['task_id'],'stage2PercentComplete')
    print(rows)
    return JsonResponse({'datas':rows}, safe=False)

@accept_websocket
def job_manage(request):
    r = redis.Redis(host=config.host2, port=config.port2, db=config.db6, decode_responses=True)
    if not request.is_websocket():
        now=datetime.now()


        # db = MySQLdb.connect("10.39.211.198", "root", "password", "busycell", charset='utf8')
        # cursor=db.cursor(DictCursor)
        # # job_query_sql = """
        # #           select a.task_id,a.name,a.create_date,b.state,b.runtime
        # #           from app_job a inner join djcelery_taskstate b on a.task_id=b.task_id order by a.create_date desc"""
        # job_query_sql="""select a.task_id,a.name,a.create_date,b.status as state,a.runtime,b.result,b.traceback
        #           from app_job a inner join celery_taskmeta b on a.task_id=b.task_id order by a.create_date desc"""
        # cursor.execute(job_query_sql)
        # rows = cursor.fetchall()
        # cursor.close()
        # db.close()

        def add_extra(job):
            # state=job.taskmeta.status
            print(hasattr(job, 'taskmeta'))
            if not hasattr(job.taskmeta,'status'):
                job.taskmeta=TaskMeta(status='UNKNOWN')
            state = job.taskmeta.status
            setattr(job, 'canKill', True if state=='STARTED' else False)
            setattr(job, 'killUrl', reverse('job_kill', kwargs={'task_id': job.task_id}) if state=='STARTED' else None)
            setattr(job, 'durationFormatted', (now-job.create_date).seconds if state=='STARTED' else job.runtime)
            setattr(job, 'stage1PercentComplete', r.hget(job.task_id, 'stage1PercentComplete') if state == 'STARTED' else 100)
            setattr(job, 'stage2PercentComplete', r.hget(job.task_id, 'stage2PercentComplete') if state == 'STARTED' else 100)
            setattr(job, 'canKill', True if state == 'STARTED' else False)
        jobs = Job.objects.all()
        list(map(add_extra, jobs))
        rows = serializer(
            jobs,
            datetime_format='string',
            foreign=True,
            # output_type='json'
        )

        # for row in rows:
        #     row['canKill']=True if row['state']=='STARTED' else False
        #     row['killUrl']=reverse('job_kill', kwargs={'task_id': row['task_id']}) if row['state']=='STARTED' else None
        #     row['durationFormatted']=(now-row['create_date']).seconds if row['state']=='STARTED' else row['runtime']
        #     row['stage1PercentComplete'] = r.hget(row['task_id'], 'stage1PercentComplete') if row['state']=='STARTED' else 100
        #     row['stage2PercentComplete'] = r.hget(row['task_id'], 'stage2PercentComplete') if row['state'] == 'STARTED' else 100
            # if row['state'] == 'STARTED':
            #     row['stage1PercentComplete']=r.hget(row['task_id'],'stage1PercentComplete')
            #     print(r.hget(row['task_id'],'stage1PercentComplete'))
            #     row['stage2PercentComplete']=r.hget(row['task_id'],'stage2PercentComplete')
        print(rows)
        return JsonResponse({'datas':rows}, safe=False)
    else:
        for message in request.websocket:
            print('message:',message)
            message = None if message == None else message.decode('utf-8')
            # if message == None:
            #     request.websocket.close()
                # break
            # if message == b'start':
            if message=='start':
            #     db = MySQLdb.connect("10.39.211.198", "root", "password", "busycell", charset='utf8')
            #     db.autocommit(True)
            #     # cursor = db.cursor(DictCursor)
            #     job_query_sql = """select a.task_id,a.name,a.create_date,b.status as state,a.runtime,b.result,b.traceback
            #                       from app_job a inner join celery_taskmeta b on a.task_id=b.task_id order by a.create_date desc"""
            #     # i=1
            # # while True:
            #     # i=i+1
            #     now = datetime.now()
            #     cursor = db.cursor(DictCursor)
            #     cursor.execute(job_query_sql)
            #     rows = cursor.fetchall()
            #     cursor.close()
            #     # db.close()
            #     for row in rows:
            #         row['canKill'] = True if row['state'] == 'STARTED' else False
            #         row['killUrl'] = reverse('job_kill', kwargs={'task_id': row['task_id']}) if row[
            #                                                                                         'state'] == 'STARTED' else None
            #         row['durationFormatted'] = (now - row['create_date']).seconds if row['state'] == 'STARTED' else row[
            #             'runtime']
            #         row['stage1PercentComplete'] = r.hget(row['task_id'], 'stage1PercentComplete') if row[
            #                                                                                               'state'] == 'STARTED' else 100
            #         row['stage2PercentComplete'] = r.hget(row['task_id'], 'stage2PercentComplete') if row[
            #                                                                                               'state'] == 'STARTED' else 100
            #         # if row['state'] == 'STARTED':
            #         #     row['stage1PercentComplete']=r.hget(row['task_id'],'stage1PercentComplete')
            #         #     print(r.hget(row['task_id'],'stage1PercentComplete'))
            #         #     row['stage2PercentComplete']=r.hget(row['task_id'],'stage2PercentComplete')

                now = datetime.now()
                def add_extra(job):
                    # state = job.taskmeta.status
                    if not hasattr(job.taskmeta, 'status'):
                        job.taskmeta = TaskMeta(status='UNKNOWN')
                    state = job.taskmeta.status
                    setattr(job, 'canKill', True if state == 'STARTED' else False)
                    setattr(job, 'killUrl',
                            reverse('job_kill', kwargs={'task_id': job.task_id}) if state == 'STARTED' else None)
                    setattr(job, 'durationFormatted',
                            (now - job.create_date).seconds if state == 'STARTED' else job.runtime)
                    setattr(job, 'stage1PercentComplete',
                            r.hget(job.task_id, 'stage1PercentComplete') if state == 'STARTED' else 100)
                    setattr(job, 'stage2PercentComplete',
                            r.hget(job.task_id, 'stage2PercentComplete') if state == 'STARTED' else 100)
                    setattr(job, 'canKill', True if state == 'STARTED' else False)

                jobs = Job.objects.all()
                list(map(add_extra, jobs))
                rows = serializer(
                    jobs,
                    datetime_format='string',
                    foreign=True,
                    # output_type='json'
                )

                msg=json.dumps(rows,cls=ComplexEncoder)
                print(msg)
                request.websocket.send(msg.encode('utf-8'))  # 发送消息到客户端
                # time.sleep(2)
                # if message == None:
                #     request.websocket.close()
                #     break
            # else:
            #     pass

def get_job_result(request):
    def file_iterator(file_name, chunk_size=512):
        with open(file_name,mode='rb') as f:
            while True:
                c = f.read(chunk_size)
                if c:
                    yield c
                else:
                    break
    task_id = request.GET.get('task_id')
    format = request.GET.get('format')
    asyncresult=AsyncResult(task_id)
    if asyncresult.successful():
        the_file_name=asyncresult.result+'.'+format
        response = StreamingHttpResponse(file_iterator(the_file_name))
        response['Content-Type'] = 'application/octet-stream'
        response['Content-Disposition'] = 'attachment;filename="{0}"'.format('result.'+format)
        return response

def job_kill(request,task_id):
    print('kill:',task_id)
    res={}
    try:
        # with current_app.default_connection() as connection:
        #     revoke(task_id,connection=connection, terminate=True)
        revoke(task_id, terminate=True)
        taskstate = TaskState.objects.filter(task_id=task_id).first()
        taskmeta = TaskMeta.objects.filter(task_id=task_id).first()
        print(taskmeta)
        if taskstate:
            if taskstate.state!='REVOKED':
                taskstate.state='REVOKED'
                taskstate.save()
        if taskmeta:
            if taskmeta.status!='REVOKED':
                taskmeta.status='REVOKED'
                taskmeta.save()
        job = Job.objects.filter(task_id=task_id).first()
        if job:
            # job.runtime = (datetime.now() - job.create_date).seconds
            # job.save()
            channel=job.id
            print('channel:',channel)
            redis_helper = RedisHelper(channel)
            redis_helper.public('killed')
            res['detail']='消息发送成功！'
        else:
            res['detail'] = 'job %s 已被删除！'
        res['result'] = 'success'
    except Exception as e:
        print(e)
        res['result'] = 'fail'
        res['detail'] = str(e)
    return JsonResponse(res)


    # r = redis.Redis(host=config.host2, port=config.port2, db=config.db6,decode_responses=True)
    # # jobs=Job.objects.all()
    # jobs=list(Job.objects.values())
    # for job in jobs:
    #     task_id=job['task_id']
    #     key = 'celery-task-meta-' + task_id
    #     result=r.get(key)
    #     # print(result)
    #     status = json.loads(result)['status']
    #     job['status']=status
    # # datas=serializers.serialize('json',jobs)
    # datas=json.dumps(jobs,cls=ComplexEncoder)
    # return HttpResponse(datas,content_type='application/json')
    # # return JsonResponse({},safe=False)

# @require_websocket
# def analyze_websocket(request):
#     for message in request.websocket:
#         print(message)
#         obj = RedisHelper(message)
#         redis_sub = obj.subscribe()
#
#         while True:
#             msg = redis_sub.parse_response()
#             # print('接收：', [x.decode('utf-8') for x in msg])
#             msg = msg[2]
#             request.websocket.send(msg)  # 发送消息到客户端

def add1(request):
    add.delay(40,80)
    # r = analyzeTask.delay(546)
    # df_busy_info = r.id
    # print(df_busy_info)
    # print(r.result)
    # print(r.get(timeout=6))
    return HttpResponse('h')