#建立评价指标，评价模型
import numpy as np
from users.recommend.myRecommend import *

#召回率描述有多少比例的用户—物品评分记录包含在最终的推荐列表中，而准确率描述最终的推荐列表中有多少比例是发生过的用户—物品评分记录

def recall(trainset,testset,N,recommend,movie_matrix,user_matrix):
    all=0
    rec_list=0
    for user,movie in trainset.items():
        if user_matrix[user] in testset.keys():
            test_u=testset[user_matrix[user]]
        train_u=recommend(user)

        train_id=[]

        for i in range(len(movie_matrix)):
            if movie_matrix[i] in train_u:
                train_id.append(i)

        for item in train_id:
            if item in test_u:
                rec_list+=1
        all+=len(test_u)
    return rec_list/all

def precision(trainset,testset,N,recommend,movie_matrix,user_matrix):
    all=0
    rec_list=0
    # rs=recommendSys()
    for user,movie in trainset.items():
        if user_matrix[user] in testset.keys():
            test_u=testset[user_matrix[user]]
        train_u=recommend(user)

        train_id=[]

        for i in range(len(movie_matrix)):
            if movie_matrix[i] in train_u:
                train_id.append(i)

        for item in train_id:
            if item in test_u:
                rec_list+=1
        all+=N
    return  rec_list/all


#覆盖率表示最终的推荐列表中包含多大比例的物品
def coverage(trainset,testset,N,recommend,movie_matrix,user_matrix):
    allitems=set()
    coverageitems=set()
    for user,movie in trainset.items():
        for item in trainset[user].keys():
            allitems.add(item)

        train_u=recommend(user)

        train_id=[]

        for i in range(len(movie_matrix)):
            if movie_matrix[i] in train_u:
                train_id.append(i)

        for item in train_id:
            coverageitems.add(item)
    return len(coverageitems)/len(coverageitems)


#新颖度：用推荐列表中物品的平均流行度度量推荐结果的新颖度，如果推荐出的物品都很热门，说明推荐的新颖度较低，否则说明推荐结果比较新颖。
def popularity(trainset,testset,N,recommend,movie_matrix,user_matrix):
    #得到流行表
    popularitems=dict()
    for user,movie in trainset.items():
        for item in movie.keys():
            if item not in popularitems:
                popularitems[item]=0
            popularitems[item]+=1
    res=0
    n=0
    for user,movie in trainset.items():
        train_u=recommend(user)

        train_id=[]

        for i in range(len(movie_matrix)):
            if movie_matrix[i] in train_u:
                train_id.append(i)

        for item in train_id:
            res+=np.log(1+popularitems[item])
            n+=1

    return res/n

