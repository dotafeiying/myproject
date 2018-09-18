# -*- coding: utf-8 -*-
import requests
import pandas as pd
import MySQLdb
from functools import reduce



def translate(ggpoints):
    points_str=';'.join([str(x[0])+','+str(x[1]) for x in ggpoints])
    # points='114.21892734521,29.575429778924;115.21892734521,39.575429778924'
    url="http://api.map.baidu.com/geoconv/v1/?coords="+points_str+"&from=1&to=5&ak=GygcEB3Vw3NOYWlHDq1KOOz2vI0C2ZCG"
    r = requests.get(url)
    result=r.json()
    status=result['status']
    # print('status:',status)
    if status==0:
        pots=result['result']
        # print(len(pots))
        points=[(p['x'],p['y']) for p in pots]
    else:
        print('status:', status)
        message=result['message']
        print('message:',message)
        num=len(ggpoints)
        points=[(message,message) for i in range(num)]
    return points

def baidu_translate(datas):
    #分组，100个一组
    datas_group=[datas[i:i+100] for i in range(0,len(datas),100)]
    # print(datas_group)
    pts=list(map(translate,datas_group))
    # print('result:',pts)
    #转换后的经纬度数组
    pts=reduce(lambda x,y:x+y,pts)
    # print('result1:',pts)
    return pts

def df_baidu_translate(df,columns=['lng','lat'],addcolumns=['BDlng','BDlat']):
    point_list = df[columns].values
    BDpoint_list = baidu_translate(point_list)
    df_bdpoint = pd.DataFrame(BDpoint_list, columns=addcolumns)
    # print(df_bdpoint.shape)
    # print(df_bdpoint[df_bdpoint['BDlng'].isnull()])
    # print(df_bdpoint[df_bdpoint.isnull().values == True])
    df[addcolumns[0]] = df_bdpoint[addcolumns[0]].values
    df[addcolumns[1]] = df_bdpoint[addcolumns[1]].values
    return df

if __name__=='__main__':
    # conn=MySQLdb.connect("10.39.211.198", "root", "password", "test", charset='utf8')
    # cursor = conn.cursor()
    # cursor.execute('select longitude,latitude from summary_http limit 203')
    # datas=cursor.fetchall()
    # print(datas)
    # # datas=[(110.920639, 31.650983), (110.4775, 31.401944), (110.236944, 31.373055)]
    # cursor.close()
    # conn.commit()
    # conn.close()

    df = pd.read_csv('test.csv')
    # print(df)
    df=df_baidu_translate(df,columns=['mr_longitude', 'mr_latitude'])
    print(df)
    # df = df[['mr_longitude', 'mr_latitude']].iloc[:20]
    # datas=df.values
    # points=baidu_translate(datas)
    # print(points)
    # df1=pd.DataFrame(points,columns=['BDlng','BDlat'])
    # print(df1)
    # df2=pd.concat([df,df1],axis=1)
    # df['bdlng','bdlat']=points
    # print(df2)