# 电影推荐系统项目实现（二）——基于用户行为数据的电影推荐系统

## 项目目的

对https://github.com/JaniceWuo/MovieRecommend 推荐系统的改进

1、增加隐语义模型和基于图的算法

2、同时增加评价指标准确率、召回率、覆盖度、新颖度来评估算法性能

3、针对冷启动问题采用最热门推荐

## 数据集

电影表（从网上搜集，自行导入）

包含电影对应的编号（不是从1开始）、电影名称、电影海报的地址

用户评分表（由django模型生成）

包含编号（主键），评分的用户id，电影对应的编号，打分

用户表（（由django模型中注册页面自动生成）

包含编号（主键），密码，最后一次登录时间，用户名称，邮箱等其他信息

## 实验设计

### 总体实验设计流程

如下：

![](https://cdn.jsdelivr.net/gh/iamxpf/pageImage/images/20201125110624.png)

（1）推荐系统的数据可以分为显性反馈数据（有用户的评分）和隐性反馈数据集（没有评分，只看用户的行为），项目的数据集为显性反馈数据

（2）划分数据集

在训练集上建立用户—兴趣模型，并在测试集上对用户行为进行预测，统计出相应的评测指标

（3）确定评价指标

### 评价指标

（1）准确率：描述最终的推荐列表中有多少比例是发生过的用户—物品评分记录

（2）召回率：描述有多少比例的用户—物品评分记录包含在最终的推荐列表中

（3）覆盖率：反映了推荐算法发掘长尾的能力，覆盖率越高，说明推荐算法越能够将长尾中的物品推荐给用户

（4）新颖度：列表中物品的平均流行度度量推荐结果的新颖度。如果推荐出的物品都很热门，说明推荐的新颖度较低，否则说明推荐结果比较新颖

### 算法原理

（1）基于邻域的算法

（2）隐语义模型

（3）基于图的随机游走算法

