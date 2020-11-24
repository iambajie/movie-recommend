#recommend2的评价指标
import numpy as np
class Metric():
    def __init__(self, train, test, recommend):
        self.train = train
        self.test = test
        self.recommend = recommend
        self.recs = self.getRec()

    #为每个用户推荐
    def getRec(self):
        recs={}
        for user in self.test:
            rank=self.recommend(user)
            recs[user]=rank
        return recs


    #召回率 有多少比例的用户—物品评分记录包含在最终的推荐列表中
    def recall(self):
        rec_len=0
        all=0
        for user in self.test:
            test_items=set(self.test[user])
            rank=self.recommend(user)

            for it,rate in rank:
                if it in test_items:
                    rec_len+=1
            all+=len(test_items)

        return rec_len/all

    #准确率  最终的推荐列表中有多少比例是发生过的用户—物品评分记录
    def precision(self):
        rec_len = 0
        all = 0
        for user in self.test:
            test_items = set(self.test[user])
            rank = self.recommend(user)

            for it, rate in rank:
                if it in test_items:
                    rec_len += 1
            all += len(rank)

        return rec_len / all


    def eval(self):
        metric = {'Precision': self.precision(),
                  'Recall': self.recall()}
        print('Metric:', metric)
        return metric