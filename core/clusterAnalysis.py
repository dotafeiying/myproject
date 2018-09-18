# from sklearn.cluster import DBSCAN,KMeans
#
#
# def run(data,radius=300):
#     res={}
#     # 默认参数 epsilon=0.001, min_samples=200
#     epsilon = radius / 100000
#     # epsilon = 0.003
#     min_samples = 100
#     db = DBSCAN(eps=epsilon, min_samples=min_samples)
#     # eps表示两个向量可以被视作为同一个类的最大的距离
#     # min_samples表示一个类中至少要包含的元素数量,如果小于这个数量,那么不构成一个类
#     y_pred = db.fit_predict(data)
#     # print(y_pred)
#     # df_user_info['label'] = y_pred
#     n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)  # 获取分簇的数目
#     if n_clusters_<1:
#         model = KMeans(n_clusters=1, random_state=0)
#         model.fit(data)
#         centroid = model.cluster_centers_
#         res['point']=