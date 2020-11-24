#解决冷启动问题
#热门推荐
import numpy as np
import sys

class clodStartRS(object):
    def __init__(self):
        self.initialset = {}
        self.popolar=3

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

    def recommend(self):
        # 获取当前最热门的电影
        N = dict()
        for user, related_mv in self.initialset.items():
            for mv, rating in related_mv.items():
                N[mv] += 1

        print(N)
        rec_list=sorted(N.items(),key=lambda j:j[1],reverse=True)[:self.popolar]
        print(rec_list)

        return rec_list