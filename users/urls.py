from django.conf.urls import url,re_path
from . import views

app_name = 'users'
urlpatterns = [
    url(r'^register/', views.register, name='register'),
    # url(r'^showmessage/', views.showmessage, name='showmessage'),
    url(r'^showmessage?(\d+)$', views.showmessage, name='showmessage'),

    # re_path('howmessage/(?P\d+)/', views.showmessage, name='showmessage')

]