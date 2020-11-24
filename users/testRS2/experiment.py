#对recommend2进行测试
from users.testRS2.Metric import *
from users.testRS2.Dataset import *
import numpy as np
import os
class Experiment(object):
    def __init__(self, M, N, fp, rt='popularTag'):
        self.M = M
        self.N = N
        self.fp = fp
        self.rt = rt
        self.alg = {'popularTag': popularTag, 'popularTag2': popularTag2, \
                    'popularTag3': popularTag3, 'improveTag': improveTag}

    # 定义单次实验
    def worker(self, train, test):
        recommend = self.alg[self.rt](train, self.N)
        metric = Metric(train, test, recommend) #评价指标
        return metric.eval()

    # 多次实验取平均
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0,
                   'Coverage': 0, 'Diversity': 0,
                   'Popularity': 0}
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test = dataset.splitdataset(self.M, ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}): {}'.format( \
            self.M, self.N, metrics))

    # 基于热门标签的推荐

def popularTag(train, N):
    # 计算user_tag和tag_item的个数
    user_tag = {}
    tag_item = {}
    for user in train:
        user_tag.setdefault(user, {})
        for item in train[user]:
            for tag in train[user][item]:
                user_tag[user].setdefault(tag, 0)
                user_tag[user][tag] += 1

                tag_item.setdefault(tag, {})
                tag_item[tag].setdefault(item, 0)
                tag_item[tag][item] += 1

    def recommend(user):
        if user not in user_tag:#添加约束
            return []
        item_score = {}
        seen_item = set(train[user])
        for tag in user_tag[user]:
            for item in tag_item[tag]:
                if item in seen_item:
                    continue
                item_score.setdefault(item, 0)
                item_score[item] += user_tag[user][tag] * tag_item[tag][item]

        return sorted(item_score.items(), key=lambda j: j[1], reverse=True)[:N]

    return recommend

# 为热门标签加上惩罚项：统计打过标签的不同用户数
def popularTag2(train,N):
    tag_pops = {}
    # 计算user_tag和tag_item的个数
    user_tag = {}
    tag_item = {}
    for user in train:
        user_tag.setdefault(user, {})
        for item in train[user]:
            for tag in train[user][item]:
                user_tag[user].setdefault(tag, 0)
                user_tag[user][tag] += 1

                tag_item.setdefault(tag, {})
                tag_item[tag].setdefault(item, 0)
                tag_item[tag][item] += 1
                if tag not in tag_pops:
                    tag_pops[tag] = set()
                else:
                    tag_pops[tag].add(user)

    tag_pop = {}
    for tag, user in tag_pops.items():
        tag_pop[tag] = len(user)
    # tag_pop = {k: len(v) for k, v in tag_pops.items()}


    def recommend(user):
        if user not in user_tag:#添加约束
            return []
        item_score = {}
        seen_item = set(train[user])

        for tag in user_tag[user]:
            for item in tag_item[tag]:
                if item in seen_item:
                    continue
                item_score.setdefault(item, 0)
                item_score[item] += user_tag[user][tag] * tag_item[tag][item] / tag_pop[tag]

        return sorted(item_score.items(), key=lambda j: j[1], reverse=True)[:N]

    return recommend

    # 为热门物品加上惩罚项：统计打过标签的不同用户数，物品的不同用户数

def popularTag3(train,N):
    tag_pops = {}
    item_pops = {}
    # 计算user_tag和tag_item的个数
    user_tag = {}
    tag_item = {}
    for user in train:
        user_tag.setdefault(user, {})
        for item in train[user]:
            # if item not in item_pops:
            #     item_pops[item] = set()
            # else:
            #     item_pops[item].add(user)
            if item not in item_pops:
                item_pops[item] = 0
            item_pops[item] += 1
            for tag in train[user][item]:
                user_tag[user].setdefault(tag, 0)
                user_tag[user][tag] += 1

                tag_item.setdefault(tag, {})
                tag_item[tag].setdefault(item, 0)
                tag_item[tag][item] += 1
                if tag not in tag_pops:
                    tag_pops[tag] = set()
                else:
                    tag_pops[tag].add(user)

    tag_pop = {}
    for tag, user in tag_pops.items():
        tag_pop[tag] = len(user)


    def recommend(user):
        if user not in user_tag:#添加约束
            return []
        item_score = {}
        seen_item = set(train[user])


        for tag in user_tag[user]:
            for item in tag_item[tag]:
                if item in seen_item:
                    continue
                item_score.setdefault(item, 0)
                item_score[item] += user_tag[user][tag] * tag_item[tag][item] / tag_pop[tag] / item_pops[item]

        return sorted(item_score.items(), key=lambda j: j[1], reverse=True)[:N]

    return recommend

# 基于标签改进的推荐
def improveTag(train, N,M=20):
    # 计算标签之间的相似度
    item_tag = {}
    for user in train:
        for item in train[user]:
            if item not in item_tag:
                item_tag[item] = set()
            for tag in train[user][item]:
                item_tag[item].add(tag)

    tag_cnt = {}
    tag_sim = {}
    for item in item_tag:
        for u in item_tag[item]:
            tag_sim.setdefault(u, {})
            if u not in tag_cnt:
                tag_cnt[u] = 0
            tag_cnt[u] += 1
            for v in item_tag[item]:
                if u == v:
                    continue
                tag_sim[u].setdefault(v, 0)
                tag_sim[u][v] += 1

    for u in tag_sim:
        for v in tag_sim[u]:
            tag_sim[u][v] /= np.sqrt(tag_cnt[u] * tag_cnt[v])

    # 为每个用户扩展标签
    user_tag = {}
    for user in train:
        user_tag.setdefault(user, {})
        for item in train[user]:
            for tag in train[user][item]:
                user_tag[user].setdefault(tag, 0)
                user_tag[user][tag] += 1

    expand_tag = {}
    for user in user_tag:
        if (len(user_tag[user]) >= M):
            expand_tag[user] = user_tag[user]
            continue
        seen_tag = set(user_tag[user])
        expand_tag[user] = {}
        for tag in user_tag[user]:
            for v in tag_sim[tag]:
                if v in seen_tag:
                    continue
                expand_tag[user].setdefault(v, 0)
                expand_tag[user][v] += np.sqrt(user_tag[user][tag] * tag_sim[tag][v])
        expand_tag[user].update(user_tag[user])
        expand_tag[user] = dict(list(sorted(expand_tag[user].items(), key=lambda j: j[1], reverse=True)[:M]))

    tag_item = {}
    for user in train:
        for item in train[user]:
            for tag in train[user][item]:
                tag_item.setdefault(tag, {})
                tag_item[tag].setdefault(item, 0)
                tag_item[tag][item] += 1

    def recommend(user):
        if user not in user_tag:#添加约束
            return []
        item_score = {}
        seen_item = set(train[user])

        for tag in expand_tag[user]:
            for item in tag_item[tag]:
                if item in seen_item:
                    continue
                item_score.setdefault(item, 0)
                item_score[item] += expand_tag[user][tag] * tag_item[tag][item]

        return sorted(item_score.items(), key=lambda j: j[1], reverse=True)[:N]

    return recommend

