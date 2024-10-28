# Create your views here.
import json
from .util.utils import create_response
from django.shortcuts import get_object_or_404, render
from django.http import Http404
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.template import loader
from .models import Question, Choice
from django.views import generic
from django.utils import timezone
from django.http import JsonResponse
# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
# from rest_framework.authtoken.models import Token
from krabs.models import User
from krabs.models import KrabsToken as Token
class IndexView(generic.ListView):
    template_name = 'krabs/index.html'
    context_object_name = 'latest_question_list'

    def get_queryset(self):
        return Question.objects.filter(
            pub_date__lte=timezone.now()
        ).order_by('-pub_date')[:5]


class DetailView(generic.DetailView):
    model = Question
    template_name = 'krabs/detail.html'

    def get_queryset(self):
        """
        Excludes any questions that aren't published yet.
        """
        return Question.objects.filter(pub_date__lte=timezone.now())


class ResultsView(generic.DetailView):
    model = Question
    template_name = 'krabs/results.html'


def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'krabs/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('krabs:results', args=(question.id,)))


# 创建的一些api接口
class LoginView(APIView):
    def get(self, request):
        data = {'message': 'Hello, this is your API response!'}
        return Response(data)

    def post(self, request):
        try:
            data = json.loads(request.body)
            username = data.get('username')
            password = data.get('password')
            # 查询用户
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                return JsonResponse({'code': 20001, 'message': '用户名或密码错误'}, status=200)

            if username == user.username and password == user.password or not user:
                # 查询用户，创建token
                user = User.objects.get(username='krabs')  # 替换为实际用户名
                # 第一个是Token对象，第二个是布尔值，代表Token是否过期
                token, flag = Token.objects.get_or_create(user=user)
                return JsonResponse(create_response(200, 20000, "登录成功", data={"token": token.key}), status=200)
            else:
                return JsonResponse({'code': 20001, 'message': '用户名或密码错误'}, status=200)
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON'}, status=400)

class UserView(APIView):
    def get(self, request):
        # 从请求体里面获取token
        token = request.query_params.get("token")
        # 在Token模型中获取User的数据
        try:
              user = Token.objects.get(key=token).user
        except Token.DoesNotExist:
            return JsonResponse(create_response(200, 20000, "未查找到Token，请重新登录", data={"isLogin": False}), status=200)
        return JsonResponse(create_response(200, 20000, "登录成功", data={"isLogin": True, "username": user.username}), status=200)

    def post(self, request):
        try:
            data = json.loads(request.body)
            username = data.get('username')
            password = data.get('password')
            # 查询用户
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                return JsonResponse({'code': 20001, 'message': '用户名或密码错误'}, status=200)

            if username == user.username and password == user.password or not user:
                # 查询用户，创建token
                user = User.objects.get(username='krabs')  # 替换为实际用户名
                # 第一个是Token对象，第二个是布尔值，代表Token是否过期
                token, flag = Token.objects.get_or_create(user=user)
                return JsonResponse(create_response(200, 20000, "登录成功", data={"token": token.key}), status=200)
            else:
                return JsonResponse({'code': 20001, 'message': '用户名或密码错误'}, status=200)
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON'}, status=400)


class RegisterView(APIView):
    def get(self, request):
        data = {'message': 'Hello, this is your API response!'}
        return Response(data)

    def post(self, request):
        try:
            data = json.loads(request.body)
            username = data.get('username')
            password = data.get('password')
            phone = data.get('phone')
            email = data.get('email')
            # 查询用户
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                user = User.objects.create(username=username, password=password, email=email, phone_number=phone)
                token, flag = Token.objects.get_or_create(user=user)
                return JsonResponse(create_response(200, 20000, "注册成功，页面即将跳转", data={"token": token.key}), status=200)
            if user is not None:
                return JsonResponse(create_response(200, 20000, "用户已经存在", data={}), status=200)
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON'}, status=400)