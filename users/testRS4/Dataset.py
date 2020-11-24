#基于上下文的推荐系统

#recommend2的数据集
import random
class Dataset():
    def __init__(self, fp,sample=10000):#取前一万条数据
        self.data = self.loaddataset(fp,sample)

    #导入数据
    def loaddataset(self,filename,sample):
        fr=open(filename)
        new_data=[]
        for lines in fr.readlines()[4:]:
            line=lines.strip().split('\t')[:3]
            # print(line)
            u,v=line
            #转换为int类型
            u=int(u)
            v=int(v)
            new_data.append((u,v))
        random.shuffle(new_data)
        return new_data[:sample]


    #划分数据集
    def splitdataset(self,m,k,seed=1):
        random.seed(seed)
        train=[]
        test=[]
        for u,v in self.data:
            if random.randint(0,m)==k:
                test.append((u,v))
            else:
                train.append((u,v))

        #将列表转换为字典形式
        def convertDict(data):
            dataset={} #当前用户指向的用户
            dataset_t={} #指向当前用户的用户
            for u,v in data:
                if u not in dataset:
                    dataset[u]=set()
                else:
                    dataset[u].add(v)
                if v not in dataset_t:
                    dataset_t[v] = set()
                else:
                    dataset_t[v].add(u)
            dataset={k:list(dataset[k]) for k in dataset}
            dataset_t={k:list(dataset_t[k]) for k in dataset_t}
            return dataset,dataset_t
        return convertDict(train),convertDict(test)[0]
