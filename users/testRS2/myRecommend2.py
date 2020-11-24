#利用用户标签数据的推荐系统
import numpy as np
import random

#剩下测试
class recommendSys2(object):
    def __init__(self):
        self.records={} #三元组：存储标签数据
        self.item_sim={} #物品的相似度
        self.user_tags={}
        self.item_tages={}
        self.tag_items={}

        self.tag_pop={}
        self.item_pop={}
        self.expand_tag={}

        self.data={}
        self.train={}
        self.test={}

        self.n_rec_item=3











    #用户u对物品i的兴趣
    def recommend(self,user):
        recommend_tag={}
        user_tags=self.user_tags
        item_tags=self.item_tages
        tagged_items=user_tags[user]

        for tag,count1 in user_tags[user].items():
            for item,count2 in item_tags[tag].items():
                if item in tagged_items:
                    continue
                if item not in recommend_tag:
                    recommend_tag[item]=count1*count2
                else:
                    recommend_tag[item]+=count1*count2

        return recommend_tag

    #改进1
    #惩罚热门标签 nb(u),标签b被多少用户使用过
    def recommend2(self,user):
        recommend_tag = {}
        user_tags=self.user_tags
        item_tags=self.item_tages
        tagged_items = user_tags[user]
        tagged={}
        for user,related_tag in user_tags.items():
            for tag,count in related_tag.items():
                tagged[tag]+=1
        for tag, count1 in user_tags[user].items():
            for item, count2 in item_tags[tag].items():
                if item in tagged_items:
                    continue
                if item not in recommend_tag:
                    recommend_tag[item] = (count1/np.log(1+tagged[tag]))*count2
                else:
                    recommend_tag[item] += (count1/np.log(1+tagged[tag]))*count2

        return recommend_tag


    #惩罚热门物品 ni(u),物品i被多少人打过标签
    def recommend2(self,user):
        recommend_tag = {}
        user_tags = self.user_tags
        item_tags = self.item_tages
        tagged_items = user_tags[user]
        tagged = {}
        for item, related_tag in item_tags.items():
            tagged[item] += 1
        for tag, count1 in user_tags[user].items():
            for item, count2 in item_tags[tag].items():
                if item in tagged_items:
                    continue
                if item not in recommend_tag:
                    recommend_tag[item] = (count1 / np.log(1 + tagged[item])) * (count2/tagged[item])
                else:
                    recommend_tag[item] += (count1 / np.log(1 + tagged[item])) * (count2/tagged[item])

        return recommend_tag

    #改进2
    #标签的相似度 nb(i)给物品i打上标签b的用户数，nb'(i)给物品i打上标签b'的用户数，求标签b和b'的相似度
    def build_tage_sim_matrix(self):
        buy={}
        tag2tag={}
        tag_sim={}

        user_tags = self.user_tags
        item_tags = self.item_tages

        for user,tag in user_tags.items():
            for i in tag.keys():
                buy[i] += 1
                tag2tag.setdefault(i,{})
                for j in tag.keys():
                    if i==j:
                        continue
                    tag2tag[i][j]+=1

        for tag1,relate_item in tag2tag.items():
            tag_sim.setdefault(tag1,{})
            for tag2,count in relate_item.items():
                tag_sim=count/(np.sqrt(buy[tag1])*np.sqrt(buy[tag2]))

        return  tag_sim



    #基于图的推荐算法

    #给用户推荐标签
    #0.给用户推荐最热门的标签
    def popularTags(self,n):
        tags={}
        for user in self.train:
            for item in self.train[user]:
                for tag in self.train[user][item]:
                    tags.setdefault(tag,0)
                    tags[tag]+=1
        return sorted(tags.items(),key=lambda j:j[1],reverse=True)[:n]

    # #1.给用户推荐物品i上最热门的标签
    # def itemPopularTags(self,user,item,item_tags,n):
    #     return sorted(item_tags[item].items(),key=lambda j:j[1],reverse=True)[:n]
    #
    # #2.给用户推荐他常使用的标签
    # def userPopularTags(self,user,item,user_tags,n):
    #     return sorted(user_tags[user].items(),key=lambda j:j[1],reverse=True)[:n]


    #推荐用户最热门的标签
    def userPopularTags2(self,user,n):
        user_tags={}
        for user in self.train:
            user_tags.setdefault(user,{})
            for item in self.train[user]:
                for tag in self.train[user][item]:
                    user_tags[user].setdefault(tag,0)
                    user_tags[user][tag]+=1
        return sorted(user_tags[user].items(),key=lambda j:j[1],reverse=True)[:n]


    #推荐物品最热门的标签
    def itemPopularTags2(self,item,n):
        item_tags={}
        for user in self.train:
            for item in self.train[user]:
                item_tags.setdefault(item,{})
                for tag in self.train[user][item]:
                    item_tags[item].setdefault(tag,0)
                    item_tags[item][tag]+=1
        return sorted(item_tags[item].items(),key=lambda j:j[1],reverse=True)[:n]


    #3.将上面两种方法结合（基于物品最热门的标签和用户最常使用的标签）
    def hybridPopularTags(self,user,item,alpha,n):
        user_tags = {}
        item_tags={}
        for user in self.train:
            user_tags.setdefault(user, {})
            for item in self.train[user]:
                item_tags.setdefault(item,{})
                for tag in self.train[user][item]:
                    user_tags[user].setdefault(tag, 0)
                    user_tags[user][tag] += 1
                    item_tags[item].setdefault(tag, 0)
                    item_tags[item][tag] += 1
        res={}
        max_value_user=max(user_tags[user].values()) #归一化
        for tag,count in user_tags[user].items():
            res[tag]+=(1-alpha)*count/max_value_user

        max_value_item=max(item_tags[item].values())
        for tag,count in item_tags.items():
            if tag not in res:
                res[tag]=alpha*count/max_value_item
            else:
                res[tag]+=alpha*count/max_value_item

        return sorted(res.items(),key=lambda j:j[1],reverse=True)[:n]
    #
    # # 基于标签的算法的评估指标
    # import numpy as np
    #
    # # 准确率
    #
    # # item[i][b]表示物品i被打上标签b的次数
    # def CosSim(item, i, j):
    #     # 物品i,j的余弦相似度
    #     res = 0
    #     ni = 0
    #     nj = 0
    #     for b, cb in item[i].items():
    #         if b in item[j].keys():
    #             res += cb * item[j][b]
    #
    #     for b, cb in item[i].items():
    #         ni += cb * cb
    #
    #     for b, cb in item[j].items():
    #         nj + +cb * cb
    #
    #     return res / (np.squrt(ni) * np.sqrt(nj))
    #
    # # 推荐系统的多样性
    # def diversity(item_tags, recommend_list):
    #     res = 0
    #     n = 0
    #     for i in recommend_list.keys():
    #         for j in recommend_list.keys():
    #             if i == j:
    #                 continue
    #             res += CosSim(item_tags, i, j)
    #             n += 1
    #
    #     return res / n

