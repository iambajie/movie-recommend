import random
import numpy as np
#设计数据格式
class Data(object):
    def __init__(self,user,item,rate,test,predict):
        self.user=user
        self.item=item
        self.rate=rate
        self.test=test
        self.predict=predict

#处理输入的数据
class Dataset(object):
    def __init__(self,fp):
        self.data=self.loadDataset(fp)

    def loadDataset(self,fp):
        fr=open(fp)
        data=[]
        for line in fr.readlines():
            line=line.strip().split('::')[:3]
            line=map(line,int)
            user,item,rate=line
            data.append(user,item,rate)

        return data

    def splitData(self,m,k,seed=1):
        random.seed(seed)
        train=[]
        test=[]
        for user,item,rate in self.data:
            if random.randint(0,m)==k:
                test.append((user,item,rate))
            else:
                train.append((user,item,rate))

        def convertToDict(data):
            dataset={}
            for user,item,rate in dataset:
                dataset.setdefault(user,{})
                dataset[user][item]=rate
            return dataset
        return convertToDict(train),convertToDict(test)



#设置评价指标RMSE
def RMSE(records):
    rmse={'train_rmse':[],'test_rmse':[]}
    for r in records:
        if r.test:
            rmse['test_rmse'].append((r.rate-r.predict)**2)
        else:
            rmse['train_rmse'].append((r.rate-r.predict)**2)

    rmse['train_rmse']=np.sqrt(np.sum(rmse['train_rmse']))/len(rmse['train_rmse'])
    rmse['test_rmse'] = np.sqrt(np.sum(rmse['test_rmse'])) / len(rmse['test_rmse'])
    return rmse



