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

    #覆盖率表示最终的推荐列表中包含多大比例的物品
    def coverage(self):
        all_items=set()
        rec_items=set()
        for user in self.train.keys():
            for item in self.train[user]:
                all_items.add(item)

        for user in self.test.keys():
            rank=self.recommend(user)
            for item,rate in rank:
                rec_items.add(item)

        return len(rec_items)/len(all_items)

    #多样性
    def diversity(self):
        item_tag={}
        for user in self.train.keys():
            for item in self.train[user]:
                if item not in item_tag:
                    item_tag[item]={}
                for tag in self.train[user][item]:
                    if tag not in item_tag:
                        item_tag[item][tag]=0
                    else:
                        item_tag[item][tag]+=1

        def Cossim(u,v):#计算两个item的相似度
            ret=0
            for tag in self.item_tages[u]:
                if tag in self.item_tages[v]:
                    ret+=self.item_tages[u][tag]*self.item_tages[v][tag]

            nu=0
            nv=0
            for tag in self.item_tages[u]:
                nu+=self.item_tages[u][tag]**2
            for tag in self.item_tages[v]:
                nv+=self.item_tages[v][tag]**2

            return ret/np.sqrt(nu+nv)
        sim=0
        all=0
        div=[]
        for user in self.test.keys():
            rank = self.recommend(user)
            for u, r1 in rank:
                for v,r2 in rank:
                    if u==v:
                        continue
                    sim+=Cossim(u,v)
                    all+=1

        sim=sim/all
        div.append(1-sim)
        return sum(div)/len(div)


    #新颖度：用推荐列表中物品的平均流行度度量推荐结果的新颖度，如果推荐出的物品都很热门，说明推荐的新颖度较低，否则说明推荐结果比较新颖。
    def popularity(self):
        item_pop={}
        for user in self.train.keys():
            for item in self.train[user]:
                if item not in item_pop:
                    item_pop[item]=1
                else:
                    item_pop[item]+=1

        n=0
        ret=0
        for user in self.test.keys():
            rank = self.recommend(user)
            for item, rate in rank:
                ret+=np.log(1+item_pop[item])
                n+=1

        return ret/n

    #导入数据
    def loaddataset(self,filename):
        fr=open(filename)
        new_data={}
        for lines in fr.readlines()[1:]:
            line=lines.strip().split('\t')[:3]
            print(line)
            for user,item,tag in line:
                if user not in new_data:
                    new_data[user]={}
                else:
                    new_data[user][item]=tag

        ret=[]
        for user,related_item in new_data.items():
            for item,tag in related_item.items():
                ret.append((user,item,tag))

    #划分数据集
    def splitdataset(self,seed,m):
        random.seed(seed)
        train=[]
        test=[]
        for user,item,tag in self.data:
            if random.randint(0,m)==1:
                test.append((user,item,tag))
            else:
                train.append((user,item,tag))



    #将列表转换为字典形式
    def convertDict(self,data):
        dataset={}
        for user,item,tag in data:
            if user not in dataset:
                dataset[user]={}
            else:
                data[user][item]=tag


    def initialSet(self):
        records=self.records
        user_tags={}
        item_tags={}
        user_items={}

        for user,item,tag in records:
            user_tags[user][tag]+=1
            item_tags[item][tag]+=1
            user_items[user][item]+=1


    #基于热门标签的推荐
    def popularTag(self):
        #计算user_tag和tag_item的个数
        user_tag={}
        tag_item={}
        for user in self.train:
            user_tag.setdefault(user,{})
            for item in self.train[user]:
                for tag in self.train[user][item]:
                    user_tag[user].setdefault(tag,0)
                    user_tag[user][tag]+=1

                    tag_item.setdefault(tag,{})
                    tag_item[tag].setdefault(item,0)
                    tag_item[tag][item]+=1

    def recommend_pop(self,user):
        item_score={}
        seen_item=self.train[user]
        user_tag=self.user_tags
        tag_items=self.tag_items
        N=self.n_rec_item

        for tag in self.user_tags[user]:
            for item in self.tag_items[tag]:
                if item in seen_item:
                    continue
                item_score.setdefault(item,0)
                item_score[item]+=user_tag[user][tag]*tag_items[tag][item]

        return sorted(item_score.items(),key=lambda j:j[1],reverse=True)[:N]

    #为热门标签加上惩罚项：统计打过标签的不同用户数
    def popularTag2(self):
        tag_pops={}
        # 计算user_tag和tag_item的个数
        user_tag = {}
        tag_item = {}
        for user in self.train:
            user_tag.setdefault(user, {})
            for item in self.train[user]:
                for tag in self.train[user][item]:
                    user_tag[user].setdefault(tag, 0)
                    user_tag[user][tag] += 1

                    tag_item.setdefault(tag, {})
                    tag_item[tag].setdefault(item, 0)
                    tag_item[tag][item] += 1
                    if tag not in tag_pops:
                        tag_pops[tag]=set()
                    else:
                        tag_pops[tag].add(user)

        tag_pop={}
        for tag,user in tag_pops:
            tag_pop[tag]=len(tag_pop[tag])

        self.tag_pop=tag_pop


    def recommend_pop2(self,user):
        item_score = {}
        seen_item = self.train[user]
        user_tag = self.user_tags
        tag_items = self.tag_items
        N = self.n_rec_item
        tag_pop=self.tag_pop

        for tag in self.user_tags[user]:
            for item in self.tag_items[tag]:
                if item in seen_item:
                    continue
                item_score.setdefault(item, 0)
                item_score[item] += user_tag[user][tag] * tag_items[tag][item]/tag_pop[tag]

        return sorted(item_score.items(), key=lambda j: j[1], reverse=True)[:N]

    #为热门物品加上惩罚项：统计打过标签的不同用户数，物品的不同用户数
    def popularTag3(self):
        tag_pops = {}
        item_pops={}
        # 计算user_tag和tag_item的个数
        user_tag = {}
        tag_item = {}
        for user in self.train:
            user_tag.setdefault(user, {})
            for item in self.train[user]:
                # if item not in item_pops:
                #     item_pops[item] = set()
                # else:
                #     item_pops[item].add(user)
                if item not in item_pops:
                    item_pops[item]=0
                else:
                    item_pops[item]+=1
                for tag in self.train[user][item]:
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
        for tag, user in tag_pops:
            tag_pop[tag] = len(tag_pop[tag])

        self.tag_pop = tag_pop

        # item_pop = {}
        # for item, user in item_pops:
        #     item_pop[item] = len(item_pop[item])

        self.item_pop = item_pops
    def recommend_pop3(self,user):
        item_score = {}
        seen_item = self.train[user]
        user_tag = self.user_tags
        tag_items = self.tag_items
        N = self.n_rec_item
        tag_pop = self.tag_pop
        item_pop=self.item_pop

        for tag in self.user_tags[user]:
            for item in self.tag_items[tag]:
                if item in seen_item:
                    continue
                item_score.setdefault(item, 0)
                item_score[item] += user_tag[user][tag] * tag_items[tag][item] / tag_pop[tag]/item_pop[item]

        return sorted(item_score.items(), key=lambda j: j[1], reverse=True)[:N]


    #基于标签改进的推荐
    def improveTag(self,M):
        # 计算标签之间的相似度
        item_tag={}
        for user in self.train:
            for item in self.train[user]:
                if item not in item_tag:
                    item_tag[item]=set()
                for tag in self.train[user][item]:
                    item_tag[item].add(tag)

        tag_cnt={}
        tag_sim={}
        for item in item_tag:
            for u in item_tag[item]:
                tag_sim.setdefault(u,{})
                if u not in tag_cnt:
                    tag_cnt[u]=0
                else:
                    tag_cnt[u]+=1
                for v in item_tag[item]:
                    if u==v:
                        continue
                    tag_sim[u].setdefault(v,0)
                    tag_sim[u][v]+=1

        for u,v in tag_sim.items():
            tag_sim[u][v]/=np.sqrt(tag_cnt[u]*tag_cnt[v])


        #为每个用户扩展标签
        user_tag={}
        for user in self.train:
            user_tag.setdefault(user,{})
            for item in self.train[user]:
                for tag in self.train[user][item]:
                    user_tag[user].setdefault(tag,0)
                    user_tag[user][tag]+=1

        expand_tag={}
        for user in user_tag.keys():
            if(len(user_tag[user])>M):
                expand_tag[user]=user_tag[user]
                continue
            seen_tag =set(user_tag[user])
            expand_tag[user]={}
            for tag in user_tag[user]:
                for v in tag_sim[tag]:
                    if v in seen_tag:
                        continue
                    expand_tag[user].setdefault(v,0)
                    expand_tag[user][v]+=np.sqrt(user_tag[user][tag]*tag_sim[tag][v])
            expand_tag[user].update(user_tag[user])
            expand_tag[user]=sorted(expand_tag[user].items(),key=lambda j:j[1],reverse=True)[:M]
        expand_tag=self.expand_tag


        user_tag = {}
        tag_item = {}
        for user in self.train:
            user_tag.setdefault(user, {})
            for item in self.train[user]:
                for tag in self.train[user][item]:
                    user_tag[user].setdefault(tag, 0)
                    user_tag[user][tag] += 1

                    tag_item.setdefault(tag, {})
                    tag_item[tag].setdefault(item, 0)
                    tag_item[tag][item] += 1

    def recommend4(self,user,N):
        item_score = {}
        seen_item = self.train[user]
        tag_items=self.tag_items
        expand_tag=self.expand_tag

        for tag in expand_tag[user]:
            for item in self.tag_items[tag]:
                if item in seen_item:
                    continue
                item_score.setdefault(item, 0)
                item_score[item] += expand_tag[user][tag] * tag_items[tag][item]

        return sorted(item_score.items(), key=lambda j: j[1], reverse=True)[:N]



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