#统一所有的推荐算法
import sys
import random
import numpy as np
import scipy.optimize as opt
import os
import os
import pymysql
import csv
import codecs

class recommendSys(object):
    def __init__(self,n):
        self.initialset={}
        self.trainset = {}
        self.testset = {}

        self.user_movie_matrix = {}
        self.movie_user_matrix={}
        self.user_sim={}
        self.movie_sim={}
        self.user_matrix = []
        self.movie_matrix = []

        self.n_rec_movie = n #推荐电影的数量
        self.k_sim_user = 10 #取10个相似的用户
        self.k_sim_movie=10 #取10个相似的电影

        # r（i，j）：表示用户j评价了电影i
        # y（i，j）：用户j对电影i的评价等级（如果用户评价过该电影）
        self.r = []
        self.y = []
        self.x = []
        self.theta = []
        self.n_features = 50
        self.l = 10
        self.movie_list = []
        self.movie_tltle={}

        self.graph = {}
        self.rank_list = {}



    #导入数据
    @staticmethod
    def loadfile(filename):
        fr=open(filename,'r',encoding='UTF-8')

        for i ,line in enumerate(fr):
            yield line.strip('\r\n')

        fr.close()
        print("load %s sucess" % filename,file=sys.stderr)

    #划分训练集和测试集
    def splitDataset(self,filename,seed,m,k):
        random.seed(seed)
        for line in self.loadfile(filename):
            id,user,movie,rating=line.split(',')
            if random.randint(0,m)==k:
                self.testset.setdefault(user,{})
                self.testset[user][movie]=rating
            else:
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = rating
        print(self.initialset)
        print(self.trainset)
        print(self.testset)



    #初始数据集
    def initial_dataset(self,filename1):
        initialset_len=0

        for lines in self.loadfile(filename1):
            id,users,movies,ratings=lines.split(',')
            # print(users,movies,ratings)
            self.initialset.setdefault(users,{})
            self.initialset[users][movies]=(ratings)
            initialset_len+=1

        # print(self.initialset)
        print("load data set sucess" , file=sys.stderr)
        print('datase=%s' % initialset_len,file=sys.stderr)

    # 得到用户-电影矩阵
    def bulid_users_movies_matrix(self, filename1, filename2):
        movie_len = 0
        user_len = 0

        for lines in self.loadfile(filename2):
            imdbid = lines.split(',')[0]
            title = lines.split(',')[1]  # 读入电影名称
            self.movie_matrix.append(imdbid)#所有电影id
            self.movie_list.append(title)
            self.movie_tltle[imdbid]=title
            movie_len += 1

        for lines in self.loadfile(filename1):
            id = lines.split(',')[0]
            self.user_matrix.append(id)#所有用户id
            user_len += 1

        print("user:%d,movie:%d" % (user_len, movie_len), file=sys.stderr)

        # self.user_movie_matrix=np.zeros((user_len,movie_len))
        dataset=self.trainset

        self.r = np.zeros((movie_len, user_len))
        self.y = np.zeros((movie_len, user_len))

        for key, value in dataset.items():
            self.user_movie_matrix.setdefault(int(key)- 1, {})
            for i in value:
                for k in range(len(self.movie_matrix)):  # 找到电影在原数组中的对应位置
                    if (self.movie_matrix[k] == i):
                        mvid = k
                        self.user_movie_matrix[int(key)- 1].setdefault(mvid, 0)
                        self.movie_user_matrix.setdefault(mvid,{})
                        break
                self.movie_user_matrix[mvid].setdefault(int(key)-1,0)
                self.user_movie_matrix[int(key)- 1][mvid] = float(dataset[key][i])
                self.movie_user_matrix[mvid][int(key)- 1] = float(dataset[key][i])

                self.r[mvid][int(key)- 1] = self.initialset[key][i]
                self.y[mvid][int(key)- 1]=1

        # print(self.user_movie_matrix)





    #冷启动

    def cold_start(self):
        # 获取当前最热门的电影
        N ={}
        for user, related_mv in self.user_movie_matrix.items():
            for mv, rating in related_mv.items():
                N.setdefault(mv,0)
                N[mv] += 1

        rankN=sorted(N.items(),key=lambda j:j[1],reverse=True)[:self.n_rec_movie]
        print(rankN)

        rec_list=[]
        for i,rating in rankN:
            rec_list.append(self.movie_matrix[i])

        return rec_list

    # 判断是否评价过电影，即新用户
    def evaluate(self, id):
        for user, mv in self.user_movie_matrix.items():
            if id == user:
                return True;

        return False


    #建立用户相似度表
    def build_users_sim_matrix(self):
        #建立电影用户倒排表
        buy={}
        user2user = {}
        movie_user_matrix = self.movie_user_matrix

        for movie,user in movie_user_matrix.items():
            for i in user.keys():
                buy.setdefault(i,0)
                buy[i]+=1
                user2user.setdefault(i,{})
                for j in user.keys():
                    if(i==j):
                        continue
                    user2user[i].setdefault(j,0)
                    user2user[i][j]+=1
                    # user_user_sim[i][j]+=np.log(1/(1+len(user))) #对用户相似度的改进

        for u1,related_u in user2user.items():
            self.user_sim.setdefault(u1,{})
            for u2,cij in related_u.items():
                self.user_sim[u1][u2]=cij/(np.sqrt(buy[u1])*np.sqrt(buy[u2])) #余弦相似度



    #计算任意两个物品之间的相似度
    def build_movies_sim_matrix(self):
        buy={}
        movie2movie={}
        user_movie_matrix=self.user_movie_matrix

        for user,movie in user_movie_matrix.items():
            for i in movie.keys():
                buy.setdefault(i,0)
                buy[i]+=1
                movie2movie.setdefault(i,{})
                for j in movie.keys():
                    if(i==j):
                        continue
                    movie2movie[i].setdefault(j, 0)
                    movie2movie[i][j]+=1
                    # movie2movie[i][j]+=1/(np.log(1+len(movie))) #用户活跃度对物品相似度的影响

        for m1,relateM in movie2movie.items():
            self.movie_sim.setdefault(m1,{})
            for m2,score in relateM.items():
                self.movie_sim[m1][m2]=score/(np.sqrt(buy[m1])*np.sqrt(buy[m2]))
                # self.movie_sim[m1][m2]/=max(self.movie_movie_sim[m1][m2]) #物品相似度的归一化


    # 推荐
    def recommend_linyu_users(self, userid):
        K=self.k_sim_user
        N=self.n_rec_movie
        user_user_sim=self.user_sim

        print('---------------')
        print(self.user_movie_matrix)
        watched_movie = self.user_movie_matrix[userid]
        rankSim = {}

        for v, sim in sorted(user_user_sim[userid].items(), key=lambda j: j[1], reverse=True)[:K]:
            for mv, rating in self.user_movie_matrix[v].items():
                if mv in watched_movie:
                    continue
                rankSim.setdefault(mv,0)

                rankSim[mv] += sim * rating

        rankN = sorted(rankSim.items(), key=lambda j: j[1], reverse=True)[:N]  # 得到了推荐的n个电影
        print(rankN)

        rec_list=[]
        for i,rating in rankN:
            rec_list.append(self.movie_matrix[i])

        return rec_list

    # 推荐系统

    def recommend_linyu_items(self,user):
        K=self.k_sim_movie
        N=self.n_rec_movie

        movie_movie_sim=self.movie_sim

        watched_movie=self.user_movie_matrix[user]
        rankSim={}
        for movie,rating in watched_movie.items():
            for related_movie,sim in sorted(movie_movie_sim[movie].items(),key=lambda j:j[1],reverse=True)[:K]:
                # if(related_movie in watched_movie):
                #     continue
                #因数据集太小，将上一句删除
                rankSim.setdefault(related_movie,0)
                rankSim[related_movie]+=sim*rating
                # rankSim[related_movie].reason[movie]=sim*rating #对推荐该电影的解释

        rankN=sorted(rankSim.items(),key=lambda j:j[1],reverse=True)[:N] #得到推荐电影的索引
        print(rankN)

        rec_list=[]
        for i,rating in rankN:
            rec_list.append(self.movie_matrix[i])

        return rec_list


    # 参数处理

    # 为了将x和theat放在一个值内进行传递
    # 合并
    def merge(self,x, theta):
        return np.concatenate((x.ravel(), theta.ravel()))

    # 分离
    def separate(self,param, n_movies, n_users, n_features):
        return param[:n_movies * n_features].reshape(n_movies, n_features), param[n_movies * n_features:].reshape(
            n_users, n_features)

    # 2.计算损失
    def cost(self,param, y, r, n_features):
        n_movies, n_users = y.shape  # y的大小是电影乘以用户
        x, theta = self.separate(param, n_movies, n_users, n_features)

        inner = np.multiply(np.dot(x, theta.T) - y, r)  # 为什么要乘以r
        inner_cost = (1 / 2) * np.sum(np.power(inner, 2))
        return inner_cost

    # 加上正则化
    def regular_cost(self,param, y, r, n_features, l=1):
        n_movies, n_users = y.shape
        x, theta = self.separate(param, n_movies, n_users, n_features)

        inner_cost = self.cost(param, y, r, n_features)

        regular = (l / 2) * np.sum(np.power(param, 2))
        return inner_cost + regular

    # 3.计算梯度
    def grad(self,param, y, r, n_features):
        n_movies, n_users = y.shape
        x, theta = self.separate(param, n_movies, n_users, n_features)

        inner = np.multiply(np.dot(x, theta.T) - y, r)
        x_grad = np.dot(inner, theta)

        theta_grad = np.dot(inner.T, x)

        return self.merge(x_grad, theta_grad)

    # 加上正则化
    def regular_grad(self,param, y, r, n_features, l=1):
        inner_grad = self.grad(param, y, r, n_features)
        regular = l * param  # param是x和theta的连接
        return inner_grad + regular


    def recommend_lfm(self,userid):
        n_movies,n_users=self.y.shape
        n_features=self.n_features
        l=self.l
        r=self.r
        y=self.y
        N=self.n_rec_movie
        movie_list=self.movie_list
        rec_list=[]
        rec_id=[]

        x_origin = np.random.standard_normal((n_movies, n_features))
        theta_origin = np.random.standard_normal((n_users, n_features))
        # param=np.concatenate((x_origin.ravel(), theta_origin.ravel()))
        param=self.merge(x_origin,theta_origin)

        y_mean=y-y.mean()

        # 对梯度进行优化
        res = opt.minimize(fun=self.regular_cost, x0=param, args=(y_mean, r, n_features, l), jac=self.regular_grad, method='TNC')
        param_train = res.x

        # 用得到的电影特征和用户偏好进行预测
        x_train, theta_train = self.separate(param_train, n_movies, n_users, n_features)

        predict = np.dot(x_train, theta_train.T)

        real_predict=predict[:,userid]+y.mean()
        # 以降序形式排列
        idx = np.argsort(real_predict)[::-1]
        print('----------------------')
        print(real_predict[idx][:N])  # 输出前10个索引对应的行的评分
        print('----------------------')


        print('推荐的电影为：')
        #获得推荐的电影的名称
        movie_list=np.array(movie_list)
        for i in movie_list[idx][:N]:
            rec_list.append(i)
            print(i)

        for imdbid,title in self.movie_tltle.items():
            if title in rec_list:
                rec_id.append(imdbid)
                print(rec_id)


        # for lines in self.loadfile(filename2):
        #     imdbid=lines.split(',')[0]
        #     title=lines.split(',')[1]#读入电影名称
        #     if title in rec_list:
        #         rec_id.append(imdbid)
        #         print(rec_id)

        return rec_id

    #基于图的随机游走算法

    #构建图矩阵 利用初始数据
    def construct_graph(self):
        user_movie_matrix=self.user_movie_matrix
        for user,related_mv in user_movie_matrix.items():
            self.graph.setdefault(user,{})
            for movie,rating in related_mv.items():
                self.graph[user].setdefault(movie, 0)

                self.graph.setdefault(movie, {})
                self.graph[movie].setdefault(user, 0)

                self.graph[user][movie]+=1
                self.graph[movie][user]+=1
        # print(('**********'))
        # print(self.graph)

    #pk算法 计算图定点的相关性
    def recommend_personalrank(self,root,alpha=0.5,max_step=50):
        Graph=self.graph
        rank=dict()
        rank={x:0 for x in Graph.keys()}
        rank[root]=1
        for k in range(max_step):
            tmp={x:0 for x in Graph.keys()}
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
        print(self.rank_list)
        rec_list = []
        for item, weight in self.rank_list:
            rec_list.append(self.movie_matrix[item])


    #推荐
    # def recommend_pk(self,userid):
    #     rec_list=[]
    #     for item, weight in self.rank_list:
    #         rec_list.append(self.movie_matrix[item])

        # for mv in self.movie_matrix:
        #     for item,weight in self.rank_list:
        #         if(item==mv):
        #             rec_list.append(item)
        #
        # for usr in self.user_matrix:
        #     for item,weight in self.rank_list:
        #         if(item==usr):
        #             rec_list.append(item)

        print('************')
        print(rec_list)
        return  rec_list



