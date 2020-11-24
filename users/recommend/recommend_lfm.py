#10-27
#使用神经网络实现协同过滤算法
#测试无误
#算法时间大约在1min

import numpy as np
import sys
import scipy.io as sio
import scipy.optimize as opt

#r（i，j）：表示用户j评价了电影i
# y（i，j）：用户j对电影i的评价等级（如果用户评价过该电影）

# x（i）：电影i的特征向量
# θ（j）：用户j对电影评价的参数

#隐语义模型  基于机器学习的方法
class lfmRS(object):
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



    #得到电影-用户矩阵
    #r（i，j）：表示用户j评价了电影i
    #y（i，j）：用户j对电影i的评价等级（如果用户评价过该电影）
    #初始化x theta
    def build_matrix(self,filename1,filename2):
        movie_len=0
        user_len=0

        movie_matrix=[]
        user_matrix=[]

        for lines in self.loadfile(filename2):
            imdbid=lines.split(',')[0]
            title=lines.split(',')[1]#读入电影名称
            self.movie_list.append(title)
            movie_matrix.append(imdbid)
            movie_len+=1


        for lines in self.loadfile(filename1):
            id=lines.split(',')[0]
            user_matrix.append(id)
            user_len+=1

        print("user:%d,movie:%d" %(user_len,movie_len),file=sys.stderr)

        self.r=np.zeros((movie_len,user_len))
        self.y = np.zeros((movie_len, user_len))

        for key,value in self.initialset.items():
            for i in value:
                for k in range(len(movie_matrix)):    #找到电影在原数组中的对应位置
                    if (movie_matrix[k] == i):
                        mvid=k
                        break
                self.r[mvid][int(key) - 1000 - 1] = self.initialset[key][i]
                self.y[mvid][int(key) - 1000 - 1]=1


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


    def recommend(self,userid,filename2):
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
        movie_list=np.array(movie_list)
        for i in movie_list[idx][:N]:
            rec_list.append(i)
            print(i)

        for lines in self.loadfile(filename2):
            imdbid=lines.split(',')[0]
            title=lines.split(',')[1]#读入电影名称
            if title in rec_list:
                rec_id.append(imdbid)
                print(rec_id)

        return rec_id
