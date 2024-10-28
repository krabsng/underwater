
# Create your models here.
import datetime
from django.utils import timezone
from django.contrib import admin
from django.contrib.auth.models import AbstractUser, Group, Permission
from django.db import models
from django.conf import settings
import uuid
from datetime import timedelta

class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')
    def __str__(self):
        return self.question_text

    @admin.display(
        boolean=True,
        ordering='pub_date',
        description="最近发布？"
    )
    def was_published_recently(self):
        now = timezone.now()
        return now - datetime.timedelta(days=1) <= self.pub_date <= now


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)
    def __str__(self):
        return self.choice_text


# 继承自抽象用户类，已有字段username，password，email
class User(AbstractUser):
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    profile_picture = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    date_joined = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    groups = models.ManyToManyField(Group, related_name='krabs_user_set', blank=True)
    user_permissions = models.ManyToManyField(Permission, related_name='krabs_user_permissions', blank=True)

    def __str__(self):
        return self.username

# 自己定义的Token
class KrabsToken(models.Model):
    # Token需要使用外键与用户绑定，用户数据被删除，该用户也会被删除
    user = models.ForeignKey(settings.TOKNE_BIND_WITH_USER_MODEL, on_delete=models.CASCADE)
    # Token键值，实际上是秘钥
    key = models.CharField(max_length=40, primary_key=True, unique=True)
    # 创建时间
    created = models.DateTimeField(auto_now_add=True)
    # 是否过期
    expired = models.BooleanField(default=False)
    # 重写的父类的save方法，该方法创建实例时自动调用
    def save(self, *args, **kwargs):
        if not self.key:
            self.key = self.generate_key()
        return super().save(*args, **kwargs)

    @staticmethod
    def generate_key():
        return str(uuid.uuid4())  # 生成一个唯一的 UUID 作为 Token

    def is_expired(self):
        return self.expired or self.created + timedelta(hours=24) < timezone.now()  # Token 24 小时后过期