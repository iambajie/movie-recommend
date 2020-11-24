import numpy as np
import random

#基于领域的协同过滤算法

#基于用户的协同过滤算法

class UserCF(object):
    def __init__(self):
        self.initialset={}
        self.trainset={}
        self.testset={}
        self.user_user_sim=dict()
        self.w=dict()
        self.n_rec_movie=3
        self.k_sim_user=10


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
            user,movie,rating=line.split(',')
            if random.randint(0,m)==k:
                self.testset.setdefault(user,{})
                self.testset[user][movie]=rating
            else:
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = rating


    #建立用户相似度表
    def build_sim_matrix(self):
        #建立电影用户倒排表
        movie2user=dict()
        for user,movie in self.trainset.items():
            for i in movie.keys():
                if i not in movie2user:
                    movie2user[i]=set()
                movie2user[i].add(user)

        user=dict()
        user_user_sim=dict()
        for movie,user in movie2user.items():
            for i in user:
                user[i]+=1
                for j in user:
                    if(i==j):
                        continue
                    user_user_sim[i][j]+=1
                    # user_user_sim[i][j]+=np.log(1/(1+len(user))) #对用户相似度的改进



        for u1,related_u in user_user_sim.items():
            for u2,cij in related_u.items():
                self.w[i][j]=cij/(np.sqrt(user[u1])*np.sqrt(user[u2]))


    #推荐
    def recommend(self,userid):
        k=self.k_sim_user
        watched_movie=self.trainset[userid]
        rank=dict()
        for v,wuv in sorted(self.w[userid].items(),key=lambda j:j[1],reverse=True)[:k]:
            for mv,rating in self.trainset[v].items():
                if mv in watched_movie:
                    continue
                rank[mv]+=wuv*rating

        N=self.n_rec_movie
        rank_n=sorted(rank.items(),key=lambda j:j[1],reverse=True)[:N] #得到了推荐的n个电影



class ItemCF(object):
    def __init__(self):
        self.initialset = {}
        self.user_movie_matrix={}
        self.movie_movie_sim={}
        self.user_matrix=[]
        self.movie_matrix=[]
        self.n_rec_movie = 3
        self.k_rec_sim=10


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

        print("load data set sucess" , file=sys.stderr)
        print('datase=%s' % initialset_len,file=sys.stderr)



    #得到用户-电影矩阵
    def build_matrix(self,filename1,filename2):
        movie_len=0
        user_len=0


        for lines in self.loadfile(filename2):
            imdbid=lines.split(',')[0]
            self.movie_matrix.append(imdbid)
            movie_len+=1

        for lines in self.loadfile(filename1):
            id=lines.split(',')[0]
            self.user_matrix.append(id)
            user_len+=1

        print("user:%d,movie:%d" %(user_len,movie_len),file=sys.stderr)

        # self.user_movie_matrix=np.zeros((user_len,movie_len))

        for key,value in self.initialset.items():
            self.user_movie_matrix.setdefault(int(key) - 1000 - 1,{})
            for i in value:
                for k in range(len(self.movie_matrix)):    #找到电影在原数组中的对应位置
                    if (self.movie_matrix[k] == i):
                        mvid=k
                        self.user_movie_matrix[int(key) - 1000 - 1].setdefault(mvid, 0)
                        break
                self.user_movie_matrix[int(key) - 1000 - 1][mvid] = float(self.initialset[key][i])



    #离线计算并保存相似度得分
    #计算任意两个物品之间的相似度
    def similarity(self):
        buy=dict()
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

                    movie2movie[i][j]+=1
                    # movie2movie[i][j]+=1/(np.log(1+len(movie))) #用户活跃度对物品相似度的影响

        for m1,relateM in movie2movie.items():
            self.movie_movie_sim.setdefault(m1,{})
            for m2,score in relateM.items():
                self.movie_movie_sim[m1][m2]=score/(np.sqrt(buy[m1])*np.sqrt(buy[m2]))
                self.movie_movie_sim[m1][m2]/=max(self.movie_movie_sim[m1][m2]) #物品相似度的归一化



    # 推荐系统

    def recommend(self,user):
        K=self.k_rec_sim
        N=self.n_rec_movie
        user_movie_matrix=self.user_movie_matrix
        movie_movie_sim=self.movie_movie_sim

        watched_movie=user_movie_matrix[user]
        rankSim={}


        for movie,rating in watched_movie.items():
            for related_movie,sim in sorted(movie_movie_sim[movie].items(),key=lambda j:j[1],reverse=True)[:K]:
                # if(related_movie in watched_movie):
                #     continue
                #因数据集太小，将上一句删除
                rankSim.setdefault(related_movie,0)
                rankSim[related_movie]+=sim*rating
                rankSim[related_movie].reason[movie]=sim*rating

        rankN=sorted(rankSim.items(),key=lambda j:j[1],reverse=True)[:N]
        print(rankN)
        rankN_list=[]
        for rec_movie,rating in rankN:
            for i in range(len(self.movie_matrix)):
                if rec_movie==i:
                    rankN_list.append(self.movie_matrix[i])
                    break;

        print(rankN_list)
        return rankN_list



