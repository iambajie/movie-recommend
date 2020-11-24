import numpy as np
import  sys
import random

random.seed(0)


#10-26重写推荐算法
#10-27 运行无误，但是算法要执行很长时间
#修改1：只做一次svd分解
#修改2：离线读取数据：预先计算所有用户和电影的相似度，减少算法时间 ---将计算相似度的值从向量变为单个的值
#svd需重做

#基于物品相似度进行推荐
class similarityRS(object):
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
                buy[i]+=movie[i]*movie[i]
                movie2movie.setdefault(i,{})
                for j in movie.keys():
                    if(i==j):
                        continue

                    movie2movie[i].setdefault(j,0)
                    movie2movie[i][j]+=movie[i]*movie[j]

        for m1,relateM in movie2movie.items():
            self.movie_movie_sim.setdefault(m1,{})
            for m2,score in relateM.items():
                self.movie_movie_sim[m1][m2]=score/(np.sqrt(buy[m1])*np.sqrt(buy[m2])) #余弦相似度



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



