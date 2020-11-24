import numpy as np
import  sys
import random

random.seed(0)


#10-26重写推荐算法
#10-27 运行无误，但是算法要执行很长时间
#修改1：只做一次svd分解
#修改2：离线读取数据：预先计算所有用户和电影的相似度，减少算法时间 未完成

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

        self.user_movie_matrix=np.zeros((user_len,movie_len))

        for key,value in self.initialset.items():
            for i in value:
                for k in range(len(self.movie_matrix)):    #找到电影在原数组中的对应位置
                    if (self.movie_matrix[k] == i):
                        mvid=k
                        break
                self.user_movie_matrix[int(key) - 1000 - 1][mvid] = self.initialset[key][i]


    # 相似度的计算

    # 将相似度控制在0~1之间
    #当输入为向量时的相似度计算
    # 欧式距离
    def edSim(inA, inB):
        return 1 / (1 + np.linalg.norm(inA - inB))  # norm 二阶范数

    # 皮尔逊系数
    def pearSim(inA, inB):
        if (len(inA) < 3):
            return 1  # 不存在三个或者更多的点，说明两个向量完全相关
        return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]

    # 余弦相似度
    def cosSim(inA, inB):
        num = inA.T * inB
        norm = np.linalg.norm(inA) * np.linalg.norm(inB)
        return 0.5 + 0.5 * (num / norm)

    #离线计算并保存相似度得分
    #计算任意两个物品之间的相似度
    def similarity(self,SimMethod=edSim):
        buy=dict()
        movie2movie={}

        for user,movie in self.user_movie_matrix:
            for i in movie.keys():
                buy.setdefault(i,0)
                buy[i]+=movie[i]*movie[i]
                movie2movie.setdefault(i,{})
                for j in movie.keys():
                    if(i==j):
                        continue

                    movie2movie[i].setdefault(j,0)
                    movie2movie[i][j]+=movie[i]*movie[j]

        for m1,relateM in movie2movie:
            for m2,score in relateM:
                self.movie_movie_sim=score/(np.sqrt(buy[m1])*np.sqrt(buy[m2])) #余弦相似度



    # 估计方法
    def standEst(dataset, user, SimMethod, item,xtransform):
        m, n = np.shape(dataset)

        simTotal = 0;
        rateSimTotal = 0

        # 遍历每个已经评级的物品
        for j in range(n):
            usrRate = dataset[user, j]
            if usrRate == 0 or j == item:
                continue
            overlap = np.nonzero(np.logical_and(dataset[:, item] > 0, dataset[:, j] > 0))[0]  # 找到评级过的两个物品

            if len(overlap) == 0:
                similarity = 0
            else:
                similarity = SimMethod(dataset[overlap, item], dataset[overlap, j])
            # print("%d and %d similarity is:%f" % (item, j, similarity))

            simTotal += similarity
            rateSimTotal += similarity * usrRate
        if simTotal == 0:
            return 0
        else:
            return rateSimTotal / simTotal

    # 利用svd提高推荐效果
    def svd(self,dataset):
        # 将原始数据映射到低维度空间
        # 选择合适的特征保留下来
        U, sigma, VT = np.linalg.svd(dataset)
        sig2 = sigma ** 2
        m = np.shape(sigma)[0]
        k = 0
        tmp_s = []
        for i in range(m):
            if np.sum(tmp_s) > (np.sum(sig2) * 0.9):
                k = i
                break;
            else:
                tmp_s = sig2[:i]
        sig4 = np.mat(np.eye(k) * sigma[:k])

        xtransform = dataset.T * U[:, :k] * np.linalg.pinv(sig4)  # 数据投影

        return xtransform


    # 指定用户对一些物品的评分
    def svdEst(dataset, user, SimMethod, item,xtransform):
        # 遍历已经评级的物品
        m, n = np.shape(dataset)

        simTotal = 0;
        rateSimTotal = 0

        for j in range(n):
            usrRate = dataset[user, j]
            if usrRate == 0 or j == item:
                continue
            similarity = SimMethod(xtransform[item, :].T, xtransform[j, :].T)
            print("%d and %d similarity is:%f" % (item, j, similarity))
            simTotal += similarity
            rateSimTotal += usrRate * similarity

        if simTotal == 0:
            return 0
        else:
            return rateSimTotal / simTotal


    # 推荐系统
    def recomend2(self, user, SimMethod=edSim, estMethod=svdEst):
        # 得到没有评级的物品
        print("----------")
        print(user)

        self.user_matrix=np.mat(self.user_matrix)

        nonest = np.nonzero(self.user_matrix[user, :] == 0)[1]  # [1]得到列向量
        print(nonest)

        if (len(nonest) == 0):
            return "you all rated"


        #执行一次svd分解
        xtransform=self.svd(self.user_matrix)
        # if estMethod=='svdEst':
        #     self.svd(self.user_matrix)


        # 将没有评级的物品按照给定估计方法进行评级
        itemScore = []
        for i in nonest:
            score = estMethod(self.user_matrix, user, SimMethod, i,xtransform)
            itemScore.append((i, score))

        sortItem=sorted(itemScore, key=lambda j: j[1], reverse=True)[:self.n_rec_movie]
        print(sortItem)

        # 得到前n个评级的物品
        return sortItem

    def recommend(self,user):
        K=self.k_rec_sim
        N=self.n_rec_movie
        user_movie_matrix=self.user_movie_matrix
        movie_movie_sim=self.movie_movie_sim

        watched_movie=user_movie_matrix[user]
        rankSim={}

        for movie,rating in watched_movie:
            for related_movie,sim in sorted(movie_movie_sim,key=lambda j:j[1],reverse=True)[:K]:
                if(related_movie in watched_movie):
                    continue

                rankSim.setdefault(related_movie,0)
                rankSim[related_movie]+=sim*rating

        rankN=sorted(rankSim,key=lambda j:j[1],reverse=True)[:N]
        rankN_list=[]
        for rec_movie,rating in rankN:
            for i in range(len(self.movie_matrix)):
                if rec_movie==i:
                    rankN_list.append(self.movie_matrix[i])
                    break;


        return rankN_list



