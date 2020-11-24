
#recommend2的数据集
import random
class Dataset():
    def __init__(self, fp):
        self.data = self.loaddataset(fp)

    #导入数据
    def loaddataset(self,filename):
        fr=open(filename)
        new_data={}
        for lines in fr.readlines()[1:]:
            line=lines.strip().split('\t')[:3]
            # print(line)
            user,item,tag=line
            #转换为int类型
            user=int(user)
            item=int(item)
            # tag=int(tag)
            if user not in new_data:
                new_data[user]={}
            else:
                new_data[user][item]=tag
            # for user,item,tag in line:
            #     if user not in new_data:
            #         new_data[user]={}
            #     else:
            #         new_data[user][item]=tag

        ret=[]
        for user,related_item in new_data.items():
            for item,tag in related_item.items():
                # ret.append((user,item,tag))
                ret.append((user,item,list(new_data[user][item])))

        return ret


    #划分数据集
    def splitdataset(self,m,k,seed=1):
        random.seed(seed)
        train=[]
        test=[]
        for user,item,tag in self.data:
            if random.randint(0,m)==k:
                test.append((user,item,tag))
            else:
                train.append((user,item,tag))

        #将列表转换为字典形式
        def convertDict(data):
            dataset={}
            for user,item,tag in data:
                if user not in dataset:
                    dataset[user]={}
                else:
                    dataset[user][item]=tag
            return dataset
        return convertDict(train),convertDict(test)
