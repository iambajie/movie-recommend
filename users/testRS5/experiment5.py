#评分预测算法 利用平均值预测用户对物品的评分
class Cluster(object):
    def __init__(self,record):
        self.group={}

    def getGroup(self,i):
        return 0


class IdCluster(object):
    def __init__(self, record):
        Cluster(self,record)

    def getGroup(self, i):
        return i

class experiment5(object):
    def __init__(self):
        pass
