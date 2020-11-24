from django.shortcuts import render, redirect
from .forms import RegisterForm
from users.models import Resulttable,Insertposter
from users.recommend.myRecommend import *
from users.evaluate import *
from users.testRS2.experiment import *
from users.testRS2.experiment2 import *
from users.testRS3.experiment3 import *
from users.testRS4.experiment4 import *
#用户注册
def register(request):
    # 只有当请求为 POST 时，才表示用户提交了注册信息
    if request.method == 'POST':
        form = RegisterForm(request.POST)

        # 验证数据的合法性
        if form.is_valid():
            # 如果提交数据合法，调用表单的 save 方法将用户数据保存到数据库
            form.save()

            # 注册成功，跳转回首页
            return redirect('/')
    else:
        # 请求不是 POST，表明用户正在访问注册页面，展示一个空的注册表单给用户
        form = RegisterForm()

    # 渲染模板
    # 如果用户正在访问注册页面，则渲染的是一个空的注册表单
    # 如果用户通过表单提交注册信息，但是数据验证不合法，则渲染的是一个带有错误信息的表单
    return render(request, 'users/register.html', context={'form': form})


#首页页面
def index(request):
    return render(request, 'users/..//index.html')

def check(request):
    return render((request, 'users/..//index.html'))
# def showregist(request):
#     pass


#10-22 可以随着不同用户名显示评价的电影
def showmessage(request,id):
    usermovieid = []
    usermovietitle = []
    print(id)
    this_id=str(id)+str(1000)
    data=Resulttable.objects.filter(userId=this_id)
    for row in data:
        usermovieid.append(row.imdbId)

    try:
        conn = get_conn()
        cur = conn.cursor()
        #Insertposter.objects.filter(userId=USERID).delete()
        for i in usermovieid:
            cur.execute('select * from mv where imdbId = %s',i) #movie的id在moviegenre3中找到电影名
            rr = cur.fetchall()
            for imdbId,title,poster in rr:
                usermovietitle.append(title)
                # print(title)

        # print(poster_result)
    finally:
        conn.close()
    # return render(request, 'users/message.html', locals())
    return render(request, 'users/message.html', {'usermovietitle':usermovietitle})


def myrecommend(request):
    rec_style = 1

    print(request.GET["userIdd"])
    USERID = int(request.GET["userIdd"])
    Insertposter.objects.filter(userId=USERID).delete()
    # selectMysql()
    read_mysql_to_csv('users/static', 'user')  # 追加数据，提高速率
    read_mysql_to_csv('users/static', 'mv')  # 追加数据，提高速率
    read_mysql_to_csv3('users/static','users_user')  # 追加数据，提高速率


    ratingfile = os.path.join('users/static', 'user.csv')
    mvfile = os.path.join('users/static', 'mv.csv')
    userfile = os.path.join('users/static', 'users_user.csv')

    N = 3
    rs = recommendSys(N)

    userid=USERID-1

    rs.initial_dataset(ratingfile)
    rs.splitDataset(ratingfile,0,8,4)

    rs.bulid_users_movies_matrix(userfile,mvfile)
    rs.construct_graph()

    #判断是否没有评价过电影
    isEvaluate=rs.evaluate(userid)
    print(isEvaluate)

    if isEvaluate=="": #fFlse
        rec_items=rs.cold_start() #使用冷启动
    else:
        if rec_style==1: #基于用户的推荐
            rs.build_users_sim_matrix()
            rec_items=rs.recommend_linyu_users(userid)

        elif rec_style==2: #基于电影的推荐
            rs.build_movies_sim_matrix()
            rec_items = rs.recommend_linyu_items(userid)

        elif rec_style == 3: #隐语义模型
            rec_items = rs.recommend_lfm(userid)

        elif rec_style == 4: #基于图的随机游走
            # rs.personalrank(0.5,userid,50)
            rec_items = rs.recommend_personalrank(userid)


    try:
        conn = get_conn()
        cur = conn.cursor()
        # Insertposter.objects.filter(userId=USERID).delete()
        for i in rec_items:
            cur.execute('select * from mv where imdbId = %s', i)
            rr = cur.fetchall()
            for imdbId, title, poster in rr:
                # print(value)         #value才是真正的海报链接
                if (Insertposter.objects.filter(title=title)):
                    continue
                else:
                    Insertposter.objects.create(userId=USERID, title=title, poster=poster)

        # print(poster_result)
    finally:
        conn.close()
    results = Insertposter.objects.filter(userId=USERID)
    if rec_style==1:
        return render(request, 'users/movieRecommend.html', {'results':results})
    else:
        return render(request, 'users/movieRecommend2.html', {'results':results})

#测试基于用户行为数据的推荐系统下的各项指标
def test(request):
    rec_style = 1

    read_mysql_to_csv('users/static', 'user')  # 追加数据，提高速率
    read_mysql_to_csv('users/static', 'mv')  # 追加数据，提高速率
    read_mysql_to_csv3('users/static','users_user')  # 追加数据，提高速率


    ratingfile = os.path.join('users/static', 'user.csv')
    mvfile = os.path.join('users/static', 'mv.csv')
    userfile = os.path.join('users/static', 'users_user.csv')

    N = 3
    rs = recommendSys(N)

    rs.initial_dataset(ratingfile)
    rs.splitDataset(ratingfile,0,8,4)

    rs.bulid_users_movies_matrix(userfile, mvfile)
    rs.construct_graph()

    dataset=rs.user_movie_matrix
    testset=rs.testset
    if rec_style == 1:  # 基于用户的推荐
        rs.build_users_sim_matrix()
        r = recall(dataset, testset, N, rs.recommend_linyu_users,rs.movie_matrix,rs.user_matrix)
        p = precision(dataset, testset, N, rs.recommend_linyu_users,rs.movie_matrix,rs.user_matrix)
        c = coverage(dataset, testset, N, rs.recommend_linyu_users,rs.movie_matrix,rs.user_matrix)
        pop = popularity(dataset, testset, N, rs.recommend_linyu_users,rs.movie_matrix,rs.user_matrix)
        print(r, p, c, pop)
        # rec_items = rs.recommend_linyu_users(userid)

    elif rec_style == 2:  # 基于电影的推荐
        rs.build_movies_sim_matrix()
        r = recall(rs.user_movie_matrix, testset, N, rs.recommend_linyu_items,rs.movie_matrix,rs.user_matrix)
        p = precision(dataset, testset, N, rs.recommend_linyu_items,rs.movie_matrix,rs.user_matrix)
        c = coverage(dataset, testset, N, rs.recommend_linyu_items,rs.movie_matrix,rs.user_matrix)
        pop = popularity(dataset, testset, N, rs.recommend_linyu_items,rs.movie_matrix,rs.user_matrix)
        print(r, p, c, pop)
        # rec_items = rs.recommend_linyu_items(userid)

    elif rec_style == 3:  # 隐语义模型
        r = recall(dataset, dataset, N, rs.recommend_lfm,rs.movie_matrix,rs.user_matrix)
        p = precision(dataset, dataset, N, rs.recommend_lfm,rs.movie_matrix,rs.user_matrix)
        c = coverage(dataset, dataset, N, rs.recommend_lfm,rs.movie_matrix,rs.user_matrix)
        pop = popularity(dataset, dataset, N, rs.recommend_lfm,rs.movie_matrix,rs.user_matrix)
        print(r, p, c, pop)
        # rec_items = rs.recommend_lfm(userid)

    elif rec_style == 4:  # 基于图的随机游走
        r = recall(dataset, dataset, N, rs.recommend_personalrank,rs.movie_matrix,rs.user_matrix)
        p = precision(dataset, dataset, N, rs.recommend_personalrank,rs.movie_matrix,rs.user_matrix)
        c = coverage(dataset, dataset, N, rs.recommend_personalrank,rs.movie_matrix,rs.user_matrix)
        pop = popularity(dataset, dataset, N, rs.recommend_personalrank,rs.movie_matrix,rs.user_matrix)
        print(r, p, c, pop)
        # rs.personalrank(0.5, userid, 50)
        # rec_items = rs.recommend_pk(userid)

    return render(request,'users/test.html',{'r':r,'p':p,'c':c,'pop':pop})

#测试基于用户标签数据的推荐系统
def test2(request):
    fp='./dataset/delicious/user_taggedbookmarks.dat'
    print(os.getcwd())
    res='sucess'
    M,N=2,10  #进行两次实验，推荐10个物品
    #基于标签的推荐系统
    # exp=Experiment(M, N, fp,rt='improveTag')
    # exp.run()

##测试有误
    #给用户推荐标签
    exp=Experiment2(M, N, fp,rt='hybridPopularTags') # popularTags   userPopularTags   itemPopularTags   hybridPopularTags
    exp.run()

    return render(request,'users/test2.html',{'res':res})

#测试基于时间上下文信息的推荐系统
def test3(request):
    print(os.getcwd())
    res='sucess'

    # popularItem  Titemcf   Tusercf   itemcf   usercf
    # 1. popularItem
    # K = 0  # 为保持一致而设置，随便填一个值
    # for site in ['www.nytimes.com', 'en.wikipedia.org']:
    #     for N in range(10, 110, 10):
    #         exp = Experiment3(K, N, site=site, rt='popularItem')
    #         exp.run()

    # 2. Titemcf
    K = 10
    for site in ['www.nytimes.com', 'en.wikipedia.org']:
        for N in range(10, 110, 10):
            exp = Experiment3(K, N, site=site, rt='usercf')#Titemcf   Tusercf   itemcf   usercf
            exp.run()

    return render(request,'users/test2.html',{'res':res})

#以上测试完成 11.20


#测试基于时间上下文信息的推荐系统
def test4(request):
    print(os.getcwd())
    res='sucess'

    M,N=2,10  #进行两次实验，推荐10个物品
    fp='./dataset/slashdot/soc-Slashdot0902.txt'
    rt_list=['Out', 'In', 'In_Out', 'In_Out2']
    for rt in rt_list:
        print(rt)
        exp = Experiment4(M, N,fp,rt)
        exp.run()

    return render(request,'users/test2.html',{'res':res})




#每个用户都给分配一个id，id从1开始
def insert(request):
    # MOVIEID = int(request.GET["movieId"])
    global USERID
    USERID = int(request.GET["userId"])
    # USERID = {{}}
    RATING = float(request.GET["rating"])
    IMDBID = int(request.GET["imdbId"])

    Resulttable.objects.create(userId=USERID, rating=RATING,imdbId=IMDBID)
    #print(USERID)
    # return HttpResponseRedirect('/')
    # return render(request, 'index.html',{'userId':USERID,'rating':RATING,'imdbId':IMDBID})
    # messages.success(request, "哈哈哈")
    return render(request, 'index.html',{'userId':USERID,'rating':RATING,'imdbId':IMDBID})


import os
import pymysql
import csv
import codecs


#连接mysql数据库
def get_conn():
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='xpf123321', db='moviesys', charset='utf8')
    return conn

def query_all(cur, sql, args):
    cur.execute(sql, args)
    return cur.fetchall()


#将数据库内所有用户的评分写入excel文件中
def read_mysql_to_csv(filename,filecsv):
    file=filename+"/"+filecsv+'.csv'
    print(file)
    with codecs.open(filename=file, mode='w', encoding='utf-8') as f:
        write = csv.writer(f, dialect='excel')
        conn = get_conn()
        cur = conn.cursor()
        cur.execute('select * from '+filecsv)
        #sql = ('select * from users_resulttable WHERE userId = 1001')
        rr = cur.fetchall()
        #results = query_all(cur=cur, sql=sql, args=None)
        for result in rr:
            #print(result)
            write.writerow(result[:])



def read_mysql_to_csv3(filename,filecsv):
    file = filename + "/" + filecsv + '.csv'
    print(file)
    with codecs.open(filename=file, mode='w', encoding='utf-8') as f:
        write = csv.writer(f, dialect='excel')
        conn = get_conn()
        cur = conn.cursor()
        cur.execute('select distinct(userId) from user')
        #sql = ('select * from users_resulttable WHERE userId = 1001')
        rr = cur.fetchall()
        #results = query_all(cur=cur, sql=sql, args=None)
        for result in rr:
            #print(result)
            write.writerow(result[:])


def read_mysql_to_csv4(filename,filecsv):
    file = filename + "/" + filecsv + '.csv'
    print(file)
    with codecs.open(filename=file, mode='w', encoding='utf-8') as f:
        write = csv.writer(f, dialect='excel')
        conn = get_conn()
        cur = conn.cursor()
        cur.execute('select distinct(userId) from users_resulttable')
        #sql = ('select * from users_resulttable WHERE userId = 1001')
        rr = cur.fetchall()
        #results = query_all(cur=cur, sql=sql, args=None)
        for result in rr:
            #print(result)
            write.writerow(result[:])


# #
# if __name__ == '__main__':
#     ratingfile2 = os.path.join('static', 'users_resulttable.csv')  # 一共671个用户
#
#     usercf = UserBasedCF()
#     userId = '1'
#     # usercf.initial_dataset(ratingfile1)
#     usercf.generate_dataset(ratingfile2)
#     usercf.calc_user_sim()
#     # usercf.evaluate()
#     usercf.recommend(userId)
#     # 给用户推荐10部电影  输出的是‘movieId’,兴趣度





