#基于上下文的推荐
import numpy as np

class recommendSys3(object):
    def __init__(self):
        self.train={}
        self.test={}

        self.alpha=0.5

    #给定时间T，物品i的流行度为
    def recentPopular(self,records,T,alpha):
        ret={}
        for user,item,t in records:
            if t>T:
                continue
            ret[item]=1/(1+alpha*(T-t))

        return ret

    #时间上下文相关的itemcf算法
    def itemcf(self):
        #计算两个物品之间的相似度
        N={}
        w={}
        alpha=self.alpha
        for user,item in self.train.items():
            for i,pi in item.items():
                N.setdefault(i,0)
                N[i]+=1
                w.setdefault(i,{})
                for j,qi in item.items():
                    if i==j:
                        continue
                        w[i].setdefault(j,0)
                    w[i][j]+=1/(1+alpha*abs(pi-qi))

        for i,wj in w.items():
            for j,p in wj.items():
                w[i][j]=p/np.sqrt(N[i]*N[j])

        return w

    #基于itemcf进行推荐
    def recommendItem(self,user,w,k,t0): #越靠近t0,权重应该越大
        ru=self.train[user]
        rank={}
        alpha=self.alpha


        for i,pi in ru.items():
            for j,qj in sorted(w[i].items(),key=lambda j:j[1],reverse=True)[:k]:
                if j in ru.keys():
                    continue

                rank[j]+=pi*qj/(1+alpha*abs(t0-ru[j]))

        return rank

    #时间上下文相关的usercf算法
    def usercf(self):
        item_users={}
        for user,item in self.train.items():
            for i,pi in item.items():
                item_users.setdefault(i,{})
                item_users[i].setdefault(user,0)

                item_users[i][user]=pi


        #计算两个用户之间的相似度
        N={}
        w={}
        alpha=self.alpha
        for item,user in item_users.items():
            for i,pi in user.items():
                N.setdefault(user,0)
                N[user]+=1
                w.setdefault(i,{})
                for j,qi in user.items():
                    w[i].setdefault(j,0)

                    w[i][j]+=1/(1+alpha*abs(pi-qi))

        for i,wi in w.items():
            for j,p in wi.items():
                w[i][j]=p/np.sqrt(N(i)*N[j])

        return w

    #推荐和用户user相近的用户
    def recommendUser(self,user,w,k,t0):
        rank={}
        seen_item=self.train[user]
        alpha=self.alpha
        for i,p in sorted(w[user].items,key=lambda j:j[1],reverse=True)[:k]:
            for j,q in self.train[i].items():
                if j in seen_item:
                    continue
                rank[i]+=p/(1+alpha*abs(t0-q))

        return rank
