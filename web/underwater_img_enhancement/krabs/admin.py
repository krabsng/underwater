from django.contrib import admin

from .models import Question, Choice, User, KrabsToken



class ChoiceInline(admin.TabularInline):
    model = Choice
    # e参数定义了在表单中默认显示的额外空白行的数量
    extra = 3

class UserInline(admin.TabularInline):
    model = User
    # e参数定义了在表单中默认显示的额外空白行的数量
    extra = 3

class QuestionAdmin(admin.ModelAdmin):
    fieldsets = [
        (None, {'fields': ['question_text']}),
        ('Date information', {'fields': ['pub_date']}),
    ]
    inlines = [ChoiceInline]
    list_display = ('question_text', 'pub_date', 'was_published_recently')
    list_filter = ['pub_date']
    search_fields = ['question_text']

class UserAdmin(admin.ModelAdmin):
    list_display = ('username', 'email')  # 根据需要修改字段
    search_fields = ['username', 'email']

class TokenAdmin(admin.ModelAdmin):
    list_display = ('user', 'created', 'expired')  # 根据需要修改字段
    search_fields = ['user']

admin.site.register(User, UserAdmin)

admin.site.register(Question, QuestionAdmin)

admin.site.register(KrabsToken, TokenAdmin)