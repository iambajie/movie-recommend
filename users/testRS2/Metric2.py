#recommend2的评价指标
#给每个用户推荐标签
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
            recs[user]={}
            for item in self.test[user]:
                rank=self.recommend(user,item)
                recs[user][item]=rank
        return recs


    #召回率 有多少比例的用户—物品评分记录包含在最终的推荐列表中
    def recall(self):
        rec_len=0
        all=0
        for user in self.test:
            for item in self.test[user]:
                test_items=set(self.test[user][item])
                rank=self.recommend(user,item)

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
            for item in self.test[user]:
                test_items = set(self.test[user][item])
                rank = self.recommend(user, item)

                for it, rate in rank:
                    if it in test_items:
                        rec_len += 1
                all += len(rank)

        return rec_len / all

    # #覆盖率表示最终的推荐列表中包含多大比例的物品
    # def coverage(self):
    #     all_items=set()
    #     rec_items=set()
    #     for user in self.train.keys():
    #         for item in self.train[user]:
    #             all_items.add(item)
    #
    #     for user in self.test.keys():
    #         for item in self.test[user]:
    #             rank=self.recommend(user,item)
    #             for item,rate in rank:
    #                 rec_items.add(item)
    #
    #     return len(rec_items)/len(all_items)
    #
    # #多样性
    # def diversity(self):
    #     item_tag={}
    #     for user in self.train.keys():
    #         for item in self.train[user]:
    #             if item not in item_tag:
    #                 item_tag[item]={}
    #             for tag in self.train[user][item]:
    #                 if tag not in item_tag:
    #                     item_tag[item][tag]=0
    #                 else:
    #                     item_tag[item][tag]+=1
    #
    #     def Cossim(u,v):#计算两个item的相似度
    #         ret=0
    #         for tag in item_tag[u]:
    #             if tag in item_tag[v]:
    #                 ret+=item_tag[u][tag]*item_tag[v][tag]
    #
    #         nu=0
    #         nv=0
    #         for tag in item_tag[u]:
    #             nu+=item_tag[u][tag]**2
    #         for tag in item_tag[v]:
    #             nv+=item_tag[v][tag]**2
    #
    #         return ret/np.sqrt(nu+nv)
    #     sim=0
    #     all=0
    #     div=[]
    #     for user in self.test.keys():
    #         rank = self.recommend(user)
    #         for u, r1 in rank:
    #             for v,r2 in rank:
    #                 if u==v:
    #                     continue
    #                 sim+=Cossim(u,v)
    #                 all+=1
    #
    #     sim=sim/all
    #     div.append(1-sim)
    #     return sum(div)/len(div)
    #
    #
    # #新颖度：用推荐列表中物品的平均流行度度量推荐结果的新颖度，如果推荐出的物品都很热门，说明推荐的新颖度较低，否则说明推荐结果比较新颖。
    # def popularity(self):
    #     item_pop={}
    #     for user in self.train.keys():
    #         for item in self.train[user]:
    #             if item not in item_pop:
    #                 item_pop[item]=1
    #             else:
    #                 item_pop[item]+=1
    #
    #     n=0
    #     ret=0
    #     for user in self.test.keys():
    #         for item in self.test[user]:
    #             rank = self.recommend(user,item)
    #             for item, rate in rank:
    #                 ret+=np.log(1+item_pop[item])
    #                 n+=1
    #
    #     return ret/n

    def eval(self):
        metric = {'Precision': self.precision(),
                  'Recall': self.recall(),
                  }
        print('Metric:', metric)
        return metric