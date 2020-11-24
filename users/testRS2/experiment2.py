#对recommend2进行测试
from users.testRS2.Metric2 import *
from users.testRS2.Dataset import *
import numpy as np
import math
import os
class Experiment2(object):
    def __init__(self, M, N, fp, rt='popularTag'):
        self.M = M
        self.N = N
        self.fp = fp
        self.rt = rt
        self.alg = {'popularTags': popularTags, 'userPopularTags': userPopularTags, \
                    'itemPopularTags': itemPopularTags,'hybridPopularTags': hybridPopularTags}

    # 定义单次实验
    def worker(self, train, test):
        recommend = self.alg[self.rt](train, self.N)
        metric = Metric(train, test, recommend) #评价指标
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


# 给用户推荐标签
# 0.给用户推荐最热门的标签
def popularTags(train,n):
    tags = {}
    for user in train:
        for item in train[user]:
            for tag in train[user][item]:
                tags.setdefault(tag, 0)
                tags[tag] += 1

    def recommend(user, item):
        return sorted(tags.items(), key=lambda j: j[1], reverse=True)[:n]

    return recommend


# #1.给用户推荐物品i上最热门的标签
# def itemPopularTags(user,item,item_tags,n):
#     return sorted(item_tags[item].items(),key=lambda j:j[1],reverse=True)[:n]
#
# #2.给用户推荐他常使用的标签
# def userPopularTags(user,item,user_tags,n):
#     return sorted(user_tags[user].items(),key=lambda j:j[1],reverse=True)[:n]


# 推荐用户最热门的标签
def userPopularTags(train,n):
    user_tags = {}
    for user in train:
        user_tags.setdefault(user, {})
        for item in train[user]:
            for tag in train[user][item]:
                user_tags[user].setdefault(tag, 0)
                user_tags[user][tag] += 1


    def recommend(user,item):
        if user not in user_tags:
            return []
        return sorted(user_tags[user].items(), key=lambda j: j[1], reverse=True)[:n]
    return recommend


# 推荐物品最热门的标签
def itemPopularTags(train,n):
    item_tags = {}
    for user in train:
        for item in train[user]:
            item_tags.setdefault(item, {})
            for tag in train[user][item]:
                item_tags[item].setdefault(tag, 0)
                item_tags[item][tag] += 1

    def recommend(user, item):
        if item not in item_tags:
            return []
        return sorted(item_tags[item].items(), key=lambda j: j[1], reverse=True)[:n]

    return recommend


# 3.将上面两种方法结合（基于物品最热门的标签和用户最常使用的标签）
def hybridPopularTags(train,n,alpha=0.5):
    # user_tags = {}
    # item_tags = {}
    # for user in train:
    #     user_tags.setdefault(user, {})
    #     for item in train[user]:
    #         item_tags.setdefault(item, {})
    #         for tag in train[user][item]:
    #             user_tags[user].setdefault(tag, 0)
    #             user_tags[user][tag] += 1
    #             item_tags[item].setdefault(tag, 0)
    #             item_tags[item][tag] += 1
    #
    # def recommend(user, item):
    #     res = {}
    #     if user in user_tags:
    #         max_value_user = max(list(user_tags[user].values())) # 归一化
    #         for tag in user_tags[user]:
    #             if tag not in res:
    #                 res[tag]=0
    #             res[tag] += (1 - alpha) * user_tags[user][tag] / max_value_user
    #     if item in item_tags:
    #         max_value_item = max(list(item_tags[item].values()))
    #         for tag in item_tags[item]:
    #             if tag not in res:
    #                 res[tag] = 0
    #             res[tag] += alpha * item_tags[item][tag] / max_value_item
    #     return sorted(res.items(), key=lambda j: j[1], reverse=True)[:n]
    #
    # return recommend

    # 统计user_tags
    user_tags = {}
    for user in train:
        user_tags[user] = {}
        for item in train[user]:
            for tag in train[user][item]:
                if tag not in user_tags[user]:
                    user_tags[user][tag] = 0
                user_tags[user][tag] += 1

    # 统计item_tags
    item_tags = {}
    for user in train:
        for item in train[user]:
            if item not in item_tags:
                item_tags[item] = {}
            for tag in train[user][item]:
                if tag not in item_tags[item]:
                    item_tags[item][tag] = 0
                item_tags[item][tag] += 1

    def recommend(user, item):
        tag_score = {}
        if user in user_tags:
            # max_user_tag =max(user_tags[user].values())
            for tag in user_tags[user]:
                if tag not in tag_score:
                    tag_score[tag] = 0
                tag_score[tag] += (1 - alpha) * user_tags[user][tag] #/ float(max_user_tag)
        if item in item_tags:
            # max_item_tag =max(user_tags[user].values())
            for tag in item_tags[item]:
                if tag not in tag_score:
                    tag_score[tag] = 0
                tag_score[tag] += alpha * item_tags[item][tag]# / float(max_item_tag)
        return list(sorted(tag_score.items(), key=lambda x: x[1], reverse=True))[:n]

    return recommend
