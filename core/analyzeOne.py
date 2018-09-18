import os,sys
import datetime,json,time,random
from math import radians, cos, sin, asin, sqrt,degrees
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from impala.dbapi import connect
from sqlalchemy import create_engine
from scipy.spatial import ConvexHull
# from sklearn.cluster import MeanShift, estimate_bandwidth
# from scipy.spatial.distance import pdist, squareform
# from sklearn import metrics

from core.redis_helper import Logger_Redis,RedisHelper
from core.public.geoconv_helper import baidu_translate,df_baidu_translate

pd.set_option('display.width', 400)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 70)

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    r = 6371.393
    dis=2*asin(sqrt(a))*r*1000.0
    return dis


def haversine(lonlat1, lonlat2):
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def analyze_one(key,enbid,cellid,radius=300,min_samples=200,K=1):
    from sklearn.cluster import DBSCAN, KMeans
    conn = connect(host='133.21.254.164', port=21050, database='hub_yuan',timeout=60)
    engine = create_engine('mysql+mysqldb://root:password@10.39.211.198:3306/busycell?charset=utf8')
    redis_helper = RedisHelper(key)
    df2 = pd.read_csv('基础信息表.csv', encoding='gbk')
    df3 = pd.read_csv('负荷表.csv', encoding='gbk',usecols=['enbid', 'cellid', 'pdcp_up_flow', 'pdcp_down_flow', 'prb_percent'])

    redis_helper.public('stage1:邻小区搜索')
    # sql = '''select enb_id as enbid,cast(cell_no as FLOAT ) as cellid,group_concat(cast(longitude as string),',') as lng_set,group_concat(cast(latitude as string),',') as lat_set
    #                 from (
    #                     select start_time,enb_id, cell_no,longitude, latitude, city_name
    #                     from lte_hd.clt_mr_all_mro_l
    #                     where enb_id={0} and cell_no=cast({1} as string) and year=2018 and month=7 and day=1 and hour=11
    #                 ) t
    #                 GROUP BY enb_id,cell_no'''.format(enbid,cellid)
    # df1 = pd.read_sql(sql, conn)
    df1=pd.DataFrame([[enbid,cellid]],columns=['enbid','cellid'],dtype=int)
    df_info = pd.merge(df2, df3, how='left', on=['enbid', 'cellid'])
    df_busy_info = pd.merge(df1,df2,how='left',on=['enbid','cellid'])
    # print(df_busy_info)
    # res_info=df_info.to_dict(orient='records')
    enbid1=df_busy_info['enbid'].values[0]
    lng1=df_busy_info['lng'].values[0]
    lat1 = df_busy_info['lat'].values[0]
    freqID1 = df_busy_info['freqID'].values[0]
    indoor = df_busy_info['indoor'].values[0]
    d = 0.3
    res={}
    # df_busy_bdpoint = pd.DataFrame(baidu_translate(df_busy_info[['lng', 'lat']].values), columns=['BDlng', 'BDlat'])
    # print(df_busy_bdpoint.shape)
    # # print(df_busy_bdpoint[df_busy_bdpoint.isnull().values == True])
    # df_busy_info['BDlng'] = df_busy_bdpoint['BDlng'].values
    # df_busy_info['BDlat'] = df_busy_bdpoint['BDlat'].values
    df_busy_info=df_baidu_translate(df_busy_info)
    res['enbinfo']=json.loads(df_busy_info.to_json(orient='records', force_ascii=False))[0]
    redis_helper.public('enbinfo|%s'%(df_busy_info.to_json(orient='records', force_ascii=False)))

    redis_helper.public('stage2:超忙分析|正在对%s个邻小区进行搜索'%df_info.shape[0])
    df_info['距离']=df_info.apply(lambda r:geodistance(lng1, lat1, r['lng'], r['lat']),axis=1)
    df_ncell_info=df_info[(df_info['距离']<d * 1000)&(df_info['enbid']!=enbid1)]
    redis_helper.public('stage2:超忙分析|，共搜索到邻小区%s个' % df_ncell_info.shape[0])
    df_ncell_info = df_baidu_translate(df_ncell_info)
    res['ncell'] =json.loads(df_ncell_info.to_json(orient='records', force_ascii=False))
    # time.sleep(2)
    # time.sleep(2)
    df_ncell_info=df_ncell_info.query('pdcp_up_flow>20 and pdcp_down_flow>80 and prb_percent>7')
    if df_ncell_info.enbid.count()==0:
        res['result']='优化方法负载均衡'
    else:
        df_ncell_info = df_ncell_info[(df_ncell_info['freqID']==freqID1) & (df_ncell_info['距离']==0)]
        if df_ncell_info.enbid.count()==0:
            res['result'] = '原站点扩载频'
        else:
            if indoor == '是':
                res['result'] = '新增室分系统'
            else:
                if freqID1 == 5:
                    res['result'] = '800M站点'


                else:
                    res['result'] = '非800M站点'
    # df_user_info = df_busy_info[['lng_set', 'lat_set']]

    redis_helper.public('stage3:关联用户信息|正在关联用户信息')
    redis_helper.public('正在查询impala表。。。')
    sql = '''select enb_id as enbid,cell_no as cellid,mr_longitude as lng,mr_latitude as lat
                            from lte_hd.clt_mr_all_mro_l
                            where year=2018 and month=8 and day=2  and enb_id={0} and cell_no="{1}" and mr_longitude is not null'''.format(
        enbid, cellid)
    df_user_info = pd.read_sql(sql, conn)[['lng', 'lat']]
    redis_helper.public('impala查询成功！')
    redis_helper.public('stage3:关联用户信息|,共关联到用户%s个' % df_user_info.shape[0])

    redis_helper.public('stage4:聚类分析')
    # redis_helper.public('stage4:聚类分析|正在利用DBSCAN算法分析用户聚集区域')
    redis_helper.public('stage4:聚类分析|分析用户聚集区域')
    df_user_info=df_user_info.dropna(axis=0)
    df_user_info.drop_duplicates(inplace=True)
    # print(df_user_info.shape)
    X = df_user_info.values
    '''DBSCAN算法的重点是选取的聚合半径参数和聚合所需指定的MinPts数目。
    在此使用球面距离来衡量地理位置的距离，来作为聚合的半径参数。
    如下实验，选取0.3公里作为密度聚合的半径参数，MinPts个数为5.'''
    # 默认参数 epsilon=0.001, min_samples=200
    epsilon=radius/100000
    # epsilon = 0.003
    min_samples = 100
    db = DBSCAN(eps=epsilon, min_samples=min_samples)
    # eps表示两个向量可以被视作为同一个类的最大的距离
    # min_samples表示一个类中至少要包含的元素数量,如果小于这个数量,那么不构成一个类
    y_pred = db.fit_predict(X)
    # print(y_pred)
    df_user_info['label']=y_pred
    n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)  # 获取分簇的数目
    # redis_helper.public('stage4:聚类分析|，获取到用户连片区域%s个'%n_clusters_)
    # redis_helper.public('stage4:聚类分析|。正在利用K-means算法计算用户聚集中心点')
    # K=2
    model = KMeans(n_clusters=K, random_state=0)
    centres_cluster=[]
    convexHull=[]
    for i in range(n_clusters_):
        print('簇 ', i, '的所有样本:')
        one_cluster = X[y_pred == i]
        print(len(one_cluster))
        # # hull.vertices 得到凸轮廓坐标的索引值，逆时针画
        # hull = ConvexHull(one_cluster).vertices.tolist()
        # # 得到凸轮廓坐标
        # result = one_cluster[hull].tolist()
        # convexHull.append({i:result})
        # print('凸轮廓坐标:',result)

        model.fit(one_cluster)
        centroid = model.cluster_centers_

        sample=model.labels_
        num_sample = np.array([len(one_cluster[sample == i]) for i in range(K)],dtype = np.int)
        index_cluster = np.ones(K,dtype = np.int) * i
        centre_point = np.c_[centroid, num_sample, index_cluster]
        centres_cluster.extend(centre_point.tolist())
        # centres_cluster.append(centre_point.tolist()[0])
        print('簇中心点:', centre_point)

        # centre_point = centroid.tolist()[0]
        # print('簇中心点:', centre_point)
        # centre_point.append(len(one_cluster))
        # centres_cluster.append(centre_point)
    if centres_cluster:
        print(centres_cluster)
        df_cluster=pd.DataFrame(centres_cluster,columns=['lng','lat','samples','index_cluster'])
        df_cluster=df_baidu_translate(df_cluster)
        res['cluster']=df_cluster[['BDlng','BDlat','samples','index_cluster']].values.tolist()

    print('经纬度异常：%s个'%df_user_info[df_user_info['lng']<10].shape[0])
    df_user_info=df_user_info[df_user_info['lng']>10]
    # df_user_info = df_user_info[(True - df_user_info['lng']<10)]
    # print(df_user_info.shape)

    # point_list=df_user_info[['lng','lat']].values
    # BDpoint_list = baidu_translate(point_list)
    # df_bdpoint = pd.DataFrame(BDpoint_list, columns=['BDlng', 'BDlat'])
    # print(df_bdpoint.shape)
    # # print(df_bdpoint[df_bdpoint['BDlng'].isnull()])
    # print(df_bdpoint[df_bdpoint.isnull().values==True])
    # df_user_info['BDlng']=df_bdpoint['BDlng'].values
    # df_user_info['BDlat'] = df_bdpoint['BDlat'].values
    df_user_info=df_baidu_translate(df_user_info)
    # df_user_info = pd.concat([df_user_info, df_bdpoint], axis=1,join_axes=[df_user_info.index])
    print(df_user_info.shape)
    # print(df_user_info[df_user_info.isnull().values == True])

    df_group = df_user_info[df_user_info['label'] != -1][['BDlng', 'BDlat', 'label']].groupby(['label'])
    for name, group in df_group:
        print(group)
        points = group[['BDlng', 'BDlat']].values
        # hull.vertices 得到凸轮廓坐标的索引值，逆时针画
        hull = ConvexHull(points).vertices.tolist()
        # 得到凸轮廓坐标
        result = points[hull].tolist()
        convexHull.append({'label': int(name), 'points': result})
        # print('凸轮廓坐标:', result)
    print('convexHull :',convexHull)
    res['convexHull'] = convexHull

    res['userinfo']=json.loads(df_user_info.to_json(orient='records', force_ascii=False))
    n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)  # 获取分簇的数目
    print('分簇的数目: %d' % n_clusters_)
    redis_helper.public('stage4:聚类分析|，将用户分为%s类' % n_clusters_)
    redis_helper.public('end')
    return res




    # for r in res_info:
    #     lng2=r['lng']
    #     lat2=r['lat']
    #     pdcp_up_flow = r['pdcp_up_flow']
    #     pdcp_down_flow = r['pdcp_down_flow']
    #     prb_percent = r['prb_percent']
    #     freqID2 = r['freqID']  # 邻区载频
    #     distance = geodistance(lng1, lat1, lng2, lat2)
    #     # print(distance)
    #     if distance < d * 1000:
    #         r['距离'] = '%.2f米' % distance
    #         data_cricle.append(r)
    #         if pdcp_up_flow > 800 and pdcp_down_flow > 800 and prb_percent > 75:
    #             r['是否高负荷'] = True
    #             # res['是否高负荷'] = True
    #             if distance == 0 and freqID2 == freqID1:
    #                 r['同站点是否可扩载频'] = False
    #                 df_busy_info['result'] = '待定'
    #                 continue
    #                 # res['同站点是否可扩载频'] = False
    #             else:
    #                 r['同站点是否可扩载频'] = True
    #                 df_busy_info['result'] = '原站点扩载频'
    #                 res['result']='原站点扩载频'
    #                 break
    #                 # return '原站点扩载频'
    #                 # res['同站点是否可扩载频'] = True
    #         else:
    #             df_busy_info['result'] = '优化方法负载均衡'
    #             r['是否高负荷'] = False
    #             res['result'] = '优化方法负载均衡'
    #             break
    #             # return '优化方法负载均衡'
    # if indoor=='是':
    #     # return '新增室分系统'
    #     res['result'] = '新增室分系统'
    # else:
    #     if freqID1 == 5:
    #         res['result'] = '非800M站点'
    #         # return '非800M站点'
    #     else:
    #         res['result'] = '800M站点'
    #         # return '800M站点'
    # res['ncell']=data_cricle
    # return res




if __name__=='__main__':
    start=datetime.datetime.now()
    result= analyze_one(1, 586837 ,52)
    print(result)
    end=datetime.datetime.now()
    print('总共耗时：',end-start)