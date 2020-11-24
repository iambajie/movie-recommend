#基于图的模型
#基于随机游走的personalrank算法
import numpy as np
import  sys


class graphRS(object):
    def __init__(self):
        self.initialset = {}
        # self.user_matrix=[]
        self.r=[]
        self.y=[]
        self.x=[]
        self.theta=[]
        self.n_features=50
        self.l=10
        self.movie_list=[]
        self.n_rec_movie = 3

        self.graph=dict()
        self.rank_list=dict()

        print('recommended movie number = %d' %
              self.n_rec_movie, file=sys.stderr)

    #导入数据
    @staticmethod
    def loadfile(filename):
        fr=open(filename,'r',encoding='UTF-8')

        for i ,line in enumerate(fr):
            yield line.strip('\r\n')

        fr.close()
        print("load %s sucess" % filename,file=sys.stderr)


    #初始数据集
    def initial_dataset(self,filename1):
        initialset_len=0

        for lines in self.loadfile(filename1):
            id,users,movies,ratings=lines.split(',')
            # print(users,movies,ratings)
            self.initialset.setdefault(users,{})
            self.initialset[users][movies]=(ratings)
            initialset_len+=1

        # for key,value in self.initialset.items():
        #     print(key,value)
        #     self.user_matrix.append(key)

        print("load data set sucess" , file=sys.stderr)
        print('datase=%s' % initialset_len,file=sys.stderr)


    #构建图矩阵 利用初始数据
    def construct_graph(self):
        for user,related_mv in self.initialset.items():
            for movie,rating in related_mv.items():
                self.graph[user][movie]+=1
                self.graph[movie][user]+=1

    #pk算法 计算图定点的相关性
    def personalrank(self,Graph,alpha,root,max_step):
        rank=dict()
        rank={x:0 for x in Graph.keys}
        rank[root]=1
        for k in range(max_step):
            tmp={x:0 for x in Graph.keys}
            for u,ev in Graph.items():
                for v,e in ev.items():
                    if v not in tmp:
                        tmp[v]=0
                    tmp[v]+=alpha*rank[u]/len(ev)  #ev:u的出度
                    if v==root:
                        tmp[root]+=1-alpha
                rank=tmp
        #得到从root走的所有节点的权重
        for v,e in Graph[root].items():
            for item,weight in rank.items():
                if v==item or item==root:
                    continue
                self.rank_list[item]=weight

        #返回没有和root产生行为的节点中，最优的几个节点
        self.rank_list=sorted(self.rank_list.items(),key=lambda j:j[1],reverse=True)[:self.n_rec_movie]


    #推荐
    def recommend(self):
        rec_list=[]
        for mv in self.movie_matrix:
            for item,weight in self.rank_list.items():
                if(item==mv):
                    rec_list.append(item)

        print(rec_list)
        print(self.rank_list)


