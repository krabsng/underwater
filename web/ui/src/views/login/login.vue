<template>
  <div class="container">
    <div class="illustration">
      <img src="@/assets/Login_img/login_left.png" alt="Illustration">
    </div>
    <div class="login-form">
      <h2 style="text-align: center">水下图像增强系统</h2>
      <!--使用v-model进行双向数据绑定-->
      <input v-model="loginForm.username" type="text" placeholder="账号" required>
      <input v-model="loginForm.password" type="password" placeholder="密码" required>
      <div class="checkbox-container">
        <input id="remember-me" v-model="rememberMe" type="checkbox" @change="updateCookie">
        <label for="remember-me">记住我</label>
      </div>
      <button ref="loginForm" @click.prevent="handleLogin" type="submit">登录</button>
      <div class="actions">
        <router-link to="/register">创建账户</router-link>
        <router-link to="/register">忘记密码？</router-link>
      </div>
    </div>
  </div>
</template>

<script>
// 引入js-cookie
import Cookies from 'js-cookie'
import { validUsername } from '@/utils/validate'
import { MessageBox, Message } from 'element-ui'

export default {
  name: 'Login',
  data() {
    const validateUsername = (rule, value, callback) => {
      if (!validUsername(value)) {
        callback(new Error('Please enter the correct user name'))
      } else {
        callback()
      }
    }
    const validatePassword = (rule, value, callback) => {
      if (value.length < 6) {
        callback(new Error('The password can not be less than 6 digits'))
      } else {
        callback()
      }
    }
    return {
      rememberMe: true,
      loginForm: {
        username: '',
        password: ''
      },
      loginRules: {
        username: [{ required: true, trigger: 'blur', validator: validateUsername }],
        password: [{ required: true, trigger: 'blur', validator: validatePassword }]
      },
      loading: false,
      passwordType: 'password',
      redirect: undefined
    }
  },
  watch: {
    $route: {
      handler: function(route) {
        this.redirect = route.query && route.query.redirect
      },
      // 这个选项表示在组件创建时立即执行一次处理函数
      immediate: true
    }
  },
  beforeCreate() {
    console.log('组件即将创建')
  },
  created() {
    const cookieValue = Cookies.get('rememberMe')
    this.rememberMe = cookieValue === 'true' // 转换为布尔值
    console.log(this.rememberMe)
    // Cookies.remove('username')
    if (this.rememberMe) {
      this.loginForm.username = Cookies.get('username')
      this.loginForm.password = Cookies.get('password')
    } else {
      this.loginForm.username = ''
      this.loginForm.password = ''
      Cookies.set('username', this.loginForm.username, { expires: 7 })// 7 天后过期
      Cookies.set('password', this.loginForm.password, { expires: 7 })// 7 天后过期
    }
  },
  beforeMount() {
    console.log('组件即将挂载')
  },
  mounted() {
    console.log('组件已挂载')
  },
  beforeUpdate() {
    console.log('组件即将更新')
  },
  updated() {
    console.log('组件已更新')
  },
  methods: {
    updateCookie() {
      // 更新 Cookie 状态
      Cookies.set('rememberMe', this.rememberMe, { expires: 7 })
      console.log(this.rememberMe)
    },
    showPwd() {
      if (this.passwordType === 'password') {
        this.passwordType = ''
      } else {
        this.passwordType = 'password'
      }
      this.$nextTick(() => {
        this.$refs.password.focus()
      })
    },
    handleLogin() {
      this.$store.dispatch('user/login', this.loginForm).then(() => {
        Message({
          message: '登录成功,页面即将跳转....' || 'Error',
          type: 'success',
          duration: 2000
        })
        Cookies.set('username', this.loginForm.username, { expires: 7 })// 7 天后过期
        Cookies.set('password', this.loginForm.password, { expires: 7 })// 7 天后过期
        this.$router.push({ path: this.redirect || '/' })
      }).catch(() => {
      })
    }
  }
}
</script>

<style scoped>

.container {
  display: flex;
  background-color: white;
  border-radius: 10px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  max-width: 900px;
  width: 100%;
}

.illustration {
  background: #eaf3ff;
  padding: 0px;
  display: flex;
  justify-content: center;
  align-items: center;
  flex: 1;
}

.illustration img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.login-form {
  padding: 40px;
  flex: 1;
  /*background-color: #3a8ee6;*/
}

h2 {
  margin-bottom: 20px;
  color: #333;
}

p {
  color: #777;
  margin-bottom: 20px;
}

input[type="text"], input[type="password"] {
  width: calc(100% - 40px);
  padding: 10px 20px;
  margin: 10px 0;
  border: 1px solid #ddd;
  border-radius: 5px;
}

.checkbox-container {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
}

.checkbox-container input {
  margin-right: 10px;
}

button {
  background-color: #4c84ff;
  color: white;
  padding: 12px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  width: 100%;
  transition: background-color 0.3s;
  margin-bottom: 20px;
}

button:hover {
  background-color: #346bd4;
}

.social-login {
  display: flex;
  justify-content: center;
  gap: 15px;
}

.social-login img {
  width: 35px;
  height: 35px;
  cursor: pointer;
}

.actions {
  display: flex;
  justify-content: space-between;
}

.actions a {
  text-decoration: none;
  color: #4c84ff;
}
</style>
