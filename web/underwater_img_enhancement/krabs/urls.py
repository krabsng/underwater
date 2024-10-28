from django.urls import path

from . import views
app_name = 'krabs'
# urlpatterns = [
#     # ex: /krabs/
#     path('', views.index, name='index'),
#     # ex: /krabs/5/
#     path('<int:question_id>/', views.detail, name='detail'),
#     # ex: /krabs/5/results/
#     path('<int:question_id>/results/', views.results, name='results'),
#     # ex: /krabs/5/vote/
#     path('<int:question_id>/vote/', views.vote, name='vote'),
# ]
urlpatterns = [
    #  api接口的类视图
    path('api/register/', views.RegisterView.as_view(), name='register'),
    path('api/user/', views.UserView.as_view(), name='user'),
    path('api/login/', views.LoginView.as_view(), name='login'),
    path('', views.IndexView.as_view(), name='index'),
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
    path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
    path('<int:question_id>/vote/', views.vote, name='vote'),
]