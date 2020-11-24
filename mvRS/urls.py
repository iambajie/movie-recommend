"""mvRS URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from users import views
# from users.views import insert
urlpatterns = [
    url(r'^admin/', admin.site.urls), #管理工具 用户名admin 密码xpf123321
    # 别忘记在顶部引入 include 函数
    url(r'^users/', include('users.urls')),  #指向users应用下的urls
    url(r'^users/', include('django.contrib.auth.urls')),
    url(r'^$', views.index, name='index'), #首页页面
    url(r'^insert/$', views.insert),#插入  修改：加入插入反馈
    url(r'^users/recommend/$', views.myrecommend),
    url(r'^users/recommend2/$', views.test2),
    url(r'^users/recommend3/$', views.test3),
    url(r'^users/recommend4/$', views.test4),
    url(r'^users/message/$', views.showmessage),
    url(r'^test/$', views.test),
]
