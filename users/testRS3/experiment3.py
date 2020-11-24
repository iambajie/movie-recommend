#基于上下文的推荐系统
from users.testRS3.Metric import *
from users.testRS3.Dataset import *
import numpy as np
import os
import time
class Experiment3(object):
    def __init__(self, K, N, site, rt='popularTag'):
        self.K = K
        self.N = N
        self.site = site
        self.rt = rt
        self.alg = {'popularItem':popularItem,'Titemcf': Titemcf, 'Tusercf': Tusercf, \
                    'itemcf': itemcf, 'usercf': usercf}

    # 定义单次实验
    def worker(self, train, test):
        recommend = self.alg[self.rt](train,self.K,self.N)
        metric = Metric(train, test, recommend) #评价指标
        return metric.eval()

    # 多次实验取平均
    def run(self):
        dataset = Dataset(self.site)
        train, test = dataset.splitData()
        metric = self.worker(train, test)
        print('Result (site={}, K={}, N={}): {}'.format( \
            self.site, self.K, self.N, metric))

#给用户推荐近期最热门的商品
def popularItem(train,k,n,alpha=1.0,beta=1.0,t0=int(time.time())):
    item_score={}
    for user in train:
        for item,t in train[user]:
            item_score.setdefault(item,0)
            item_score[item]+=1/(1+alpha*abs(t-t0))

    def recommend(user):
        seem_item=set(train[user])
        rank={}
        for item,score in item_score.items():
            if item in seem_item:
                continue
            rank[item]=score
        return sorted(rank.items(),key=lambda j:j[1],reverse=True)[:n]
    return recommend

#时间上下文相关的itemcf算法
def Titemcf(train,k,n,alpha=1.0,beta=1.0,t0=int(time.time())):
    #计算两个物品之间的相似度
    N={}
    w={}
    for user in train:
        item=train[user]
        for i in range(len(item)):
            u,pi=item[i]
            N.setdefault(u,0)
            N[u]+=1
            w.setdefault(u,{})
            for j in range(len(item)):
                if i==j:
                    continue
                v, qi = item[j]
                w[u].setdefault(v,0)
                w[u][v]+=1/(1+alpha*abs(pi-qi))

    for i,wj in w.items():
        for j,p in wj.items():
            w[i][j]/=np.sqrt(N[i]*N[j])
    #基于itemcf进行推荐
    def recommend(user): #越靠近t0,权重应该越大
        ru=set(train[user])
        rank={}
        for item,t in train[user]:
            for j,qj in sorted(w[item].items(),key=lambda j:j[1],reverse=True)[:k]:
                if j in ru:
                    continue
                rank.setdefault(j,0)
                rank[j]+=w[item][j]/(1+beta*abs(t-t0))

        return sorted(rank.items(),key=lambda j:j[1],reverse=True)[:n]
    return recommend

#时间上下文相关的usercf算法
def Tusercf(train,k,n,alpha=1.0,beta=1.0,t0=int(time.time())):
    item_users={}
    for user in train:
        for item,t in train[user]:
            item_users.setdefault(item,[])
            # item_users[item].setdefault(user,0)
            item_users[item].append((user,t)) #和train的数据组成类型保持一致

    #计算两个用户之间的相似度
    N={}
    w={}
    for item in item_users:
        user=item_users[item]
        for i in range(len(user)):
            u,pi=user[i]
            N.setdefault(u,0)
            N[u]+=1
            w.setdefault(u,{})
            for j in range(len(user)):
                if i==j:
                    continue
                v, qi = user[j]
                w[u].setdefault(v,0)

                w[u][v]+=1/(1+alpha*abs(pi-qi))

    for i,wi in w.items():
        for j,p in wi.items():
            w[i][j]/=np.sqrt(N[i]*N[j])


#推荐和用户user相近的用户
    def recommend(user):
        rank={}
        seen_item=set(train[user])
        if user in w:
            for i,p in sorted(w[user].items(),key=lambda j:j[1],reverse=True)[:k]:
                for j,t in train[i]:
                    if j in seen_item:
                        continue
                    rank.setdefault(i, 0)
                    rank[i]+=w[user][i]/(1+beta*abs(t-t0))

        return sorted(rank.items(),key=lambda j:j[1],reverse=True)[:n]
    return recommend


#不考虑时间的itemcf算法
def itemcf(train,k,n,alpha=1.0,beta=1.0,t0=int(time.time())):
    #计算两个物品之间的相似度
    N={}
    w={}
    for user in train:
        item= train[user]
        for i in range(len(item)):
            u, pi = item[i]
            N.setdefault(u,0)
            N[u]+=1
            w.setdefault(u,{})
            for j in range(len(item)):
                v, qj = item[j]
                if i==j:
                    continue
                w[u].setdefault(v,0)
                w[u][v]+=1

    for i,wj in w.items():
        for j,p in wj.items():
            w[i][j]/=np.sqrt(N[i]*N[j])
    #基于itemcf进行推荐
    def recommend(user): #越靠近t0,权重应该越大
        ru=set(train[user])
        rank={}
        for i,pi in train[user]:
            for j,qj in sorted(w[i].items(),key=lambda j:j[1],reverse=True)[:k]:
                if j in ru:
                    continue
                rank.setdefault(j,0)
                rank[j]+=qj

        return sorted(rank.items(),key=lambda j:j[1],reverse=True)[:n]
    return recommend

#不考虑时间的usercf算法
def usercf(train,k,n,alpha=1.0,beta=1.0,t0=int(time.time())):
    item_users={}
    for user in train:
        for i,pi in train[user]:
            item_users.setdefault(i,[])
            # item_users[i].setdefault(user,0)

            item_users[i].append((user,pi))


    #计算两个用户之间的相似度
    N={}
    w={}
    for item in item_users:
        user=item_users[item]
        for i in range(len(user)):
            u, pi = user[i]
            N.setdefault(u,0)
            N[u]+=1
            w.setdefault(u,{})
            for j in range(len(user)):
                v, qj = user[j]
                w[u].setdefault(v,0)

                w[u][v]+=1

    for i,wi in w.items():
        for j,p in wi.items():
            w[i][j]/=np.sqrt(N[i]*N[j])

#推荐和用户user相近的用户
    def recommend(user):
        rank={}
        seen_item=set(train[user])
        if user in w:
            for i,p in sorted(w[user].items(),key=lambda j:j[1],reverse=True)[:k]:
                for j,q in train[i]:
                    if j in seen_item:
                        continue
                    rank.setdefault(i,0)
                    rank[i]+=p
        return sorted(rank.items(),key=lambda j:j[1],reverse=True)[:n]
    return recommend

