#基于社交网络推荐好友
from users.testRS4.Metric import *
from users.testRS4.Dataset import *
import numpy as np
import os

class Experiment4(object):
    def __init__(self, M, N, fp, rt='Out'):
        self.M = M
        self.N = N
        self.fp = fp
        self.rt = rt
        self.alg = {'Out': Out, 'In': In, \
                    'In_Out': In_Out, 'In_Out2': In_Out2}

    # 定义单次实验
    def worker(self, train, test):
        recommend = self.alg[self.rt](train, self.N)
        metric = Metric(train[0], test, recommend) #评价指标
        return metric.eval()

    # 多次实验取平均
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0}
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test = dataset.splitdataset(self.M, ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}): {}'.format( \
            self.M, self.N, metrics))
#对于用户u，v，根据共同好友的比例来计算相似度


#以出度衡量 G为出度数组,GT为入度数组
def Out(train,n):
    G,GT=train

    def recommend(user):
        if user not in G:
            return []
        user_friend=set(G[user])
        user_sim={}
        for u in G[user]:
            if u not in GT:
                continue
            for v in GT[u]:
                if v!=user and v not in user_friend:
                    user_sim.setdefault(v,0)
                    user_sim[v]+=1

        for v in user_sim:
            user_sim[v]/=np.sqrt(len(G[user])*len(G[v]))

        return sorted(user_sim.items(),key=lambda j:j[1],reverse=True)[:n]
    return recommend


#以入度衡量
def In(train,n):
    G, GT = train

    def recommend(user):
        if user not in GT:
            return []
        user_friend = set(GT[user])
        user_sim = {}
        for u in GT[user]:
            if u not in G:
                continue
            for v in G[u]:
                if v != user and v not in user_friend:
                    user_sim.setdefault(v, 0)
                    user_sim[v] += 1

        for v in user_sim:
            user_sim[v] /= np.sqrt(len(GT[user]) * len(GT[v]))

        return sorted(user_sim.items(), key=lambda j: j[1], reverse=True)[:n]
    return recommend

#以u的出度衡量，v的入度衡量两者之间的相似度
def In_Out(train,n):
    G, GT = train
    def recommend(user):
        if user not in G:
            return []
        user_friend=set(G[user])
        user_sim={}
        for u in G[user]:
            if u not in G:
                continue
            for v in G[u]:
                if v!=user and v not in user_friend:
                    user_sim.setdefault(v, 0)
                    user_sim[v] += 1

        for v in user_sim:
            user_sim[v]/=len(G[user])

        return sorted(user_sim.items(), key=lambda j: j[1], reverse=True)[:n]
    return recommend

def In_Out2(train,n):
    G, GT = train
    def recommend(user):
        if user not in G:
            return []
        user_friend=set(G[user])
        user_sim={}
        for u in G[user]:
            if u not in G:
                continue
            for v in G[u]:
                if v!=user and v not in user_friend:
                    user_sim.setdefault(v, 0)
                    user_sim[v] += 1

        for v in user_sim:
            user_sim[v]/=np.sqrt(len(G[user])*len(GT[v]))

        return sorted(user_sim.items(), key=lambda j: j[1], reverse=True)[:n]
    return recommend