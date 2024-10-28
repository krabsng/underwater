import Vue from 'vue'
import Vuex from 'vuex'
import getters from './getters'
import app from './modules/app'
import settings from './modules/settings'
import krabs from './modules/krabs'
import user from './modules/user'

Vue.use(Vuex)

const store = new Vuex.Store({
  // Vuex 提供的一个配置选项，允许把 Vuex 的 store 分割成多个独立的、模块化的部分。每个模块都包含自己的 state、mutations、actions 和 getters。
  modules: {
    app,
    settings,
    user,
    krabs
  },
  // Vuex 全局的 getters，它可以访问整个 store 的状态。
  getters
})

export default store
