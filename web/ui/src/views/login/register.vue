<template>
  <div class="container">
    <h1>用户注册</h1>
    <form>
      <div class="form-group">
        <label for="username">用户名</label>
        <input type="text" id="username" v-model="loginForm.username" required>
      </div>
      <div class="form-group">
        <label for="email">电子邮箱</label>
        <input type="email" id="email" v-model="loginForm.email" required>
      </div>
      <div class="form-group">
        <label for="password">密码</label>
        <input type="password" id="password" v-model="loginForm.password" required>
      </div>
      <div class="form-group">
        <label for="phone">手机号码</label>
        <input type="tel" id="phone" v-model="loginForm.phone">
      </div>
      <button type="submit" @click="handleRegister">注册</button>
    </form>
  </div>
</template>

<script>
import { validUsername } from '@/utils/validate'
import { MessageBox, Message } from 'element-ui'

export default {
  name: 'Register',
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
        password: '',
        email: '',
        phone: ''
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
  methods: {
    handleRegister() {
      this.$store.dispatch('user/register', this.loginForm).then(() => {
        this.$router.push({ path: this.redirect || '/' })
      }).catch(() => {
      })
    }
  }
}
</script>

<style scoped>
.container {
  width: 25%; /* 增加最大宽度 */
  padding: 30px; /* 增加内边距 */
  background: #ffffff;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

h1 {
  text-align: center;
  color: #007BFF;
}

.form-group {
  margin-bottom: 15px;
}

label {
  display: block;
  margin-bottom: 5px;
}

input {
  width: 100%;
  padding: 10px;
  border: 1px solid #007BFF;
  border-radius: 4px;
  box-sizing: border-box;
}

button {
  width: 100%;
  padding: 10px;
  background-color: #007BFF;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

button:hover {
  background-color: #0056b3;
}
</style>
