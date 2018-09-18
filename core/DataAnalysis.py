import pandas as pd
import os
from decimal import Decimal
# from sklearn.cluster import KMeans,DBSCAN
import datetime,json,time,random
from math import radians, cos, sin, asin, sqrt,degrees
from impala.dbapi import connect
from sqlalchemy import create_engine
import MySQLdb
from collections import OrderedDict
from retrying import retry
import redis
from core.conf import config

from core.redis_helper import Logger_Redis,RedisHelper

pd.set_option('display.width', 400)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 70)

# def init_model(epsilon,min_samples,K=1,n_jobs=1):
#     from sklearn.cluster import KMeans,DBSCAN
#     kmeans_model=KMeans(n_clusters=K, random_state=0,n_jobs=n_jobs)
#     dbscan_model = DBSCAN(eps=epsilon, min_samples=min_samples, n_jobs=n_jobs)
#     return (kmeans_model, dbscan_model)


class Busycell_calc(object):
    def __init__(self,channel='test',file_path='test.xlsx',task_id=None,file_path_id=None,radius=300,min_samples=200,K=1,n_jobs=1):
        self.i = 0
        self.j = 0
        self.count = 0
        self.radius = radius
        self.min_samples = min_samples
        self.K = K
        self.n_jobs = n_jobs
        self.key = channel
        self.task_id = task_id
        self.file_path = file_path
        self.redis_helper=RedisHelper(self.key)
        self.conn=connect(host='133.21.254.163', port=21050,database='hub_yuan', timeout=30)
        self.r = redis.Redis(host=config.host2, port=config.port2, db=config.db6)
        self.engine = create_engine('mysql+mysqldb://root:password@10.39.211.198:3306/busycell?charset=utf8')
        self.db = MySQLdb.connect("10.39.211.198", "root", "password", "busycell", charset='utf8')
        self.kmeans_model = self.init_model[0]
        self.dbscan_model = self.init_model[1]
        # self.df1=self.get_busy_df
        self.df2=pd.read_csv('基础信息表.csv',encoding='gbk')
        self.df3=pd.read_csv('负荷表.csv',encoding='gbk',usecols=['enbid','cellid','pdcp_up_flow','pdcp_down_flow','prb_percent'])
        self.df_info=self.get_df_info
        # self.df_busy_info=self.get_df_busy_info

    def get_busy_df(self):
        start = datetime.datetime.now()
        # self.creatTable('超忙小区.xlsx')
        self.redis_helper.public('<b>【任务开始】 taskID：</b>{0}'.format(self.task_id))
        self.redis_helper.public('正在查询impala表。。。')
        df1=pd.read_excel(self.file_path, usecols=['province', 'city', 'enbid', 'cellid'])
        data = df1.apply(lambda x: '%d_%d' % (x['enbid'], x['cellid']), axis=1).values.tolist()
        data_str = str(data)
        sql1 = '''select enbid,cellid,group_concat(cast(longitude as string),',') as lng_set,group_concat(cast(latitude as string),',') as lat_set
                        from (
                            select cast(enb_id as int) as enbid, cast(cell_no as int) as cellid,longitude, latitude, city_name as city
                            from lte_hd.clt_mr_all_mro_l
                            where concat(cast(enb_id as string),'_',cell_no) in (%s) and 
                            year=2018 and month=8 and day=1 and hour=11 and longitude is not null and latitude is not null
                        ) t
                        GROUP BY enbid,cellid''' % (data_str[1:-1])
        df2 = pd.read_sql(sql1, self.conn)
        df = pd.merge(df1, df2, how='left', on=['enbid', 'cellid'])[['city', 'enbid', 'cellid','lng_set','lat_set']]
        self.count = df.shape[0]
        self.redis_helper.public('impala表查询完成!')
        end = datetime.datetime.now()
        self.redis_helper.public('impala表查询耗时 %ss' % (end - start).seconds)
        return df

        # sql = '''select city,enbid,cellid,group_concat(cast(longitude as string),',') as lng_set,group_concat(cast(latitude as string),',') as lat_set
        #         from (
        #             select start_time,enbid, cellid,longitude, latitude, city
        #             from lte_hd.clt_mr_all_mro_l m
        #             right join hub_yuan.lte_busy_cell n
        #             on m.enb_id=n.enbid and m.cell_no=cast(n.cellid as string) and m.year=2018 and m.month=8 and m.day=1 and m.hour=11
        #         ) t
        #         GROUP BY city,enbid,cellid'''
        # df5 = pd.read_sql(sql, self.conn)
        # # df5=df5[(df5['enbid']==602668) & (df5['cellid']==53)]
        # self.count = df5.shape[0]
        # print(self.count)
        # self.redis_helper.public('impala表查询完成!')
        # end = datetime.datetime.now()
        # self.redis_helper.public('impala表查询耗时 %ss' % (end - start).seconds)
        # return df5

    @property
    def init_model(self):
        from sklearn.cluster import KMeans, DBSCAN
        kmeans_model = KMeans(n_clusters=self.K, random_state=0, n_jobs=self.n_jobs)
        dbscan_model = DBSCAN(eps=self.radius/100000, min_samples=self.min_samples, n_jobs=self.n_jobs)
        return (kmeans_model, dbscan_model)

    @property
    def get_df_info(self):
        return pd.merge(self.df2, self.df3, how='left', on=['enbid', 'cellid'])

    def get_df_busy_info(self):
        df1=self.get_busy_df()
        return pd.merge(df1,self.df2,how='left',on=['enbid','cellid'])

    # @retry(stop_max_delay=1000*5)
    # def creatTable(self,path_busycell):
    #     start = datetime.datetime.now()
    #     # print(path_busycell,1)
    #     # df = pd.read_excel(path_busycell)
    #     # print(df.columns)
    #     df = pd.read_excel(path_busycell, usecols=['province', 'city', 'enbid', 'cellid'])
    #     self.count=df.shape[0]
    #     data = [tuple(d) for d in df.values.tolist()]
    #     print(str(data)[1:-1])
    #     sql_create = '''
    #         create table IF NOT EXISTS lte_busy_cell
    #         (
    #            province STRING,
    #            city STRING,
    #            enbid INT,
    #            cellid INT
    #         )
    #         ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
    #     '''
    #     sql_insert = 'insert overwrite lte_busy_cell values {}'.format(str(data)[1:-1])
    #     # conn = connect(host='133.21.254.164', port=21050, database='hub_yuan')
    #     cursor = self.conn.cursor()
    #     cursor.execute(sql_create)
    #     cursor.execute(sql_insert)
    #     end = datetime.datetime.now()
    #     self.redis_helper.public('创建impala表完成，耗时 %s'%(end-start).seconds)
    #     print('创建表耗时：', end - start)

    def parse_distance(self,row):
        self.i+=1
        # if self.i%100==0:
        self.redis_helper.public('开始搜索附近基站 %s个，完成 <b>%d%%</b>' %(self.i,self.i/self.count*100))
        # r = redis.Redis(host=config.host2, port=config.port2, db=config.db6)
        self.r.hset(self.task_id, "stage1PercentComplete", '%d'%(self.i/self.count*100))
        # r.set(self.task_id,(self.i/self.count*100))
        # time.sleep(random.random()/100)
        df_info = self.df_info
        enbid1, cellid1, freqID1 = row.enbid, row.cellid, row.freqID
        x1, y1 = row.lng, row.lat
        # print(x1,y1)
        d = 0.3
        r = 6371.393
        dlng = 2 * asin(sin(d / (2 * r)) / cos(y1))
        dlng = degrees(dlng)  # 弧度转换成角度
        dlat = d / r
        dlat = degrees(dlat)
        minlng = x1 - dlng
        maxlng = x1 + dlng
        minlat = y1 - dlat
        maxlat = y1 + dlat
        res = OrderedDict()
        res['是否高负荷'] = False
        res['同站点是否可扩载频'] = True
        res_df = df_info[
            (df_info.lng > minlng) & (df_info.lng < maxlng) & (df_info.lat > minlat) & (df_info.lat < maxlat) & (
            enbid1 != df_info.enbid)]
        # res_df = res_df.where(res_df.notnull(), None)
        data_rectangle = res_df.to_dict(orient='records')
        # print(res_df)
        # res_df=res_df.round({'lng':6,'lat':6})
        # data_rectangle = res_df.iterrows()

        data_cricle = []
        for r in data_rectangle:
            # if isinstance(r['lng'], float) and isinstance(r['lng'], float):
            r['lng'] = float(Decimal(r['lng']).quantize(Decimal('0.000000')))
            r['lat'] = float(Decimal(r['lat']).quantize(Decimal('0.000000')))
            x2, y2 = r['lng'], r['lat']
            # print(x2,y2)
            pdcp_up_flow = r['pdcp_up_flow']
            pdcp_down_flow = r['pdcp_down_flow']
            prb_percent = r['prb_percent']
            freqID2 = r['freqID'] #邻区载频
            distance = round(geodistance(x1, y1, x2, y2),2)
            if distance < d * 1000:
                r['距离'] = '%.1f米'%distance
                # r = r.where(r.notnull(), None).to_dict()
                # r=r.to_dict()
                # print(r)
                data_cricle.append(r)
                if pdcp_up_flow > 20 and pdcp_down_flow > 80 and prb_percent > 7:
                    r['是否高负荷'] = True
                    res['是否高负荷'] = True
                    if distance == 0 and freqID2 == freqID1:
                        r['同站点是否可扩载频'] = False
                        res['同站点是否可扩载频'] = False
                    else:
                        r['同站点是否可扩载频'] = True
                        # res['同站点是否可扩载频'] = True
                else:
                    r['是否高负荷'] = False
        res['data'] = data_cricle
        return res
        # return json.dumps(res, ensure_ascii=False)

    def parse_stage2(self,row):
        dbscan_model = self.dbscan_model
        kmeans_model = self.kmeans_model
        self.j += 1
        self.redis_helper.public('进行聚类分析，完成<b>%d%%</b>' % (self.j / self.count * 100))
        self.r.hset(self.task_id, "stage2PercentComplete", '%d'%(self.j / self.count * 100))
        # self.redis_helper.public('开始聚类分析第%s个' % self.j)
        # model = self.model
        res = OrderedDict()
        res['是否高负荷'] = False
        res['同站点是否可扩载频'] = True
        res['是否室分站'] = True
        lng_set = row['lng_set']
        lat_set = row['lat_set']
        # print('lng_set:',type(lng_set),lng_set)
        # data = json.loads(row['是否邻小区'])
        data=row['是否邻小区']
        indoor = row['indoor']
        isBusy = data['是否高负荷']
        isFreq = data['同站点是否可扩载频']
        freq = row['freqID']
        scene = row['scene']
        # data = pd.DataFrame({'lng': row['lng_set'].split(','), 'lat': row['lat_set'].split(',')})
        # print(data)
        # model.fit(data)
        # centroid = model.cluster_centers_
        # print(centroid)
        # centre_point = centroid.tolist()[0]
        # # print(centre_point)
        # res['centre'] = centre_point
        if isBusy:
            res['是否高负荷'] = True
            if not isFreq:
                res['同站点是否可扩载频'] = False
                if indoor == '否':
                    res['是否室分站'] = False
                    if not freq == 5:
                        if isinstance(lng_set,str) and isinstance(lat_set,str):
                            data = pd.DataFrame({'lng': row['lng_set'].split(','), 'lat': row['lat_set'].split(',')},columns=['lng','lat'])
                            # print("\033[0;31m%s\033[0m" % "输出红色字符")
                            # print(data)
                            # 默认参数 epsilon=0.001, min_samples=200
                            radius = 300
                            epsilon = radius / 100000
                            # epsilon = 0.003
                            min_samples = 100
                            y_pred = dbscan_model.fit_predict(data)
                            n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)  # 获取分簇的数目
                            if n_clusters_ < 1:
                                kmeans_model.fit(data)
                                centroid = kmeans_model.cluster_centers_
                                centres_cluster=centroid.tolist()[0]
                            else:
                                centres_cluster = []
                                for i in range(n_clusters_):
                                    print('簇 ', i, '的所有样本:')
                                    one_cluster = data[y_pred == i]
                                    kmeans_model.fit(one_cluster)
                                    centroid = kmeans_model.cluster_centers_
                                    centres_cluster.append(centroid.tolist()[0])
                            res['centre'] = centres_cluster
                            return '建站点为：'+ str(centres_cluster)


                            # model.fit(data)
                            # centroid = model.cluster_centers_
                            # print(centroid)
                            # centre_point = centroid.tolist()[0]
                            # # print(centre_point)
                            # res['centre'] = centre_point
                        # return '非800M站点'
                    else:
                        if scene == '市区':
                            return '优先考虑优化手段'
                        else:
                            if isinstance(lng_set,str) and isinstance(lat_set,str):
                                data = pd.DataFrame({'lng': row['lng_set'].split(','), 'lat': row['lat_set'].split(',')},columns=['lng','lat'])
                                # print(data)
                                # 默认参数 epsilon=0.001, min_samples=200
                                radius = 300
                                epsilon = radius / 100000
                                # epsilon = 0.003
                                min_samples = 100
                                y_pred = dbscan_model.fit_predict(data)
                                n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)  # 获取分簇的数目
                                if n_clusters_ < 1:
                                    kmeans_model.fit(data)
                                    centroid = kmeans_model.cluster_centers_
                                    centres_cluster = centroid.tolist()[0]
                                    return '新增L800M小区，建站点为：',str(centres_cluster)
                                else:
                                    centres_cluster = []
                                    for i in range(n_clusters_):
                                        print('簇 ', i, '的所有样本:')
                                        one_cluster = data[y_pred == i]
                                        kmeans_model.fit(one_cluster)
                                        centroid = kmeans_model.cluster_centers_
                                        centres_cluster.append(centroid.tolist()[0])
                                    return '用1.8G或者2.1G吸收，建站点为：'+ str(centres_cluster)

                                # model.fit(data)
                                # centroid = model.cluster_centers_
                                # centre_point = centroid.tolist()[0]
                                # # print(centre_point)
                                # res['密集点'] = centre_point
                            # return '800M站点'
                else:
                    return '新增室分系统或采用有源天线系统'
            else:
                return '原站点扩载频'
        else:
            return '优化方法负载均衡'
        # return json.dumps(res, ensure_ascii=False)
        # return res

    def run(self):
        df_busy_info = self.get_df_busy_info()
        engine = self.engine
        self.redis_helper.public('stage1：邻区搜索')
        df_busy_info['是否邻小区'] = df_busy_info.apply(self.parse_distance, axis=1)
        self.redis_helper.public('stage2：聚类分析')
        df_busy_info['result'] = df_busy_info.apply(self.parse_stage2, axis=1)
        df_busy_info['n_cell'] = df_busy_info.apply(lambda x:x['是否邻小区']['data'], axis=1)
        df_busy_info.drop(['是否邻小区'], axis=1, inplace=True)
        df_busy_info.drop(['lng_set', 'lat_set'], axis=1, inplace=True)
        now = datetime.datetime.now()
        df_busy_info['finish_time'] = now
        df_busy_info_copy = df_busy_info.copy()

        # df_busy_info.to_excel('result.xlsx')
        self.redis_helper.public('正在导入mysql..')
        print('导入mysql..')
        # df_busy_info['是否邻小区']=df_busy_info.apply(lambda x:json.dumps(x['是否邻小区'],ensure_ascii=False),axis=1)
        # df_busy_info['结果'] = df_busy_info.apply(lambda x: json.dumps(x['结果'], ensure_ascii=False),axis=1)
        # df_busy_info['是否邻小区']=df_busy_info['是否邻小区'].map(lambda x: json.dumps(x, ensure_ascii=False))

        # 保存到文件
        timestr = now.strftime("%Y%m%d%H%M%S")
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        download_dir = os.path.join(BASE_DIR, 'download')
        filename = str(os.path.split(self.file_path)[1].split('.')[0]) +'_'+ timestr
        # filename = str(os.path.basename(self.file_path).split('.')[0]) +'_'+ timestr
        # download_url = download_dir + filename
        download_url=os.path.join(download_dir,filename)
        print(download_dir)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        df_busy_info_copy.to_excel(download_url+ '.xlsx',index=None)
        df_busy_info_copy.to_csv(download_url+ '.csv',index=None)
        print('已保存到本地excel')

        # df_busy_info_copy['是否邻小区'] = df_busy_info_copy['是否邻小区'].map(lambda x: json.dumps(x, ensure_ascii=False))
        df_busy_info['result']=df_busy_info['result'].map(lambda x: json.dumps(x, ensure_ascii=False))
        df_busy_info['n_cell'] = df_busy_info['n_cell'].map(lambda x: json.dumps(x, ensure_ascii=False))
        # df_busy_info['lng_set'] = df_busy_info['lng_set'].map(lambda x: json.dumps(x, ensure_ascii=False))
        # df_busy_info['lat_set'] = df_busy_info['lat_set'].map(lambda x: json.dumps(x, ensure_ascii=False))
        df_busy_info.to_sql('busycell', con=engine, if_exists='append')
        df_btsinfo=df_busy_info[['city','enbid','cellid','cellname','freqID','scene','indoor']]
        df_btsinfo.drop_duplicates(inplace=True)
        df_btsinfo = df_btsinfo.where(df_btsinfo.notnull(), None)
        rows=df_btsinfo.values.tolist()
        cursor = self.db.cursor()
        sql = "insert ignore into btsinfo (city,enbid,cellid,cellname,freqID,scene,indoor) VALUES(%s,%s,%s,%s,%s,%s,%s)"
        try:
            cursor.executemany(sql, rows)
        except Exception as e:
            self.db.rollback()
            print("执行MySQL: %s 时出错：%s" % (sql, e))
        self.db.commit()
        cursor.close()
        self.db.close()

        self.redis_helper.public('导入mysql成功!')
        self.redis_helper.public('end')
        return df_busy_info,download_url

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    r = 6371.393
    dis=2*asin(sqrt(a))*r*1000.0
    return dis





# busycell=Busycell_calc()

# def run():
#     df_busy_info=busycell.df_busy_info
#     engine=busycell.engine
#     df_busy_info['是否邻小区'] = df_busy_info.apply(parse_distance, axis=1)
#     print(df_busy_info)
#     df_busy_info['结果'] = df_busy_info.apply(parse_stage2, axis=1)
#     df_busy_info.drop(['lng_set', 'lat_set'], axis=1, inplace=True)
#     now = datetime.datetime.now()
#     df_busy_info['日期'] = now
#     df_busy_info.to_excel('result.xlsx')
#     print('导入mysql..')
#     df_busy_info.to_sql('busycell', con=engine, if_exists='append')


if __name__=='__main__':
    start=datetime.datetime.now()
    busycell = Busycell_calc()
    busycell.run()
    end=datetime.datetime.now()
    print('总共耗时：',end-start)

