<template>
  <div id="app" :style="parentContainerStyle">
    <router-view />
  </div>
</template>

<script>
import { mapState, mapActions } from 'vuex'
export default {
  name: 'App',
  computed: {
    // 使用 mapState 将 Vuex 中模块的 state 映射到当前组件的计算属性中
    // 使用 mapState 将 state 中的 parentContainerStyle 映射为计算属性
    ...mapState({
      parentContainerStyle: state => state.krabs.parentContainerStyle
    })
    // // 或者使用模块名字符串
    // ...mapState('krabs', ['parentContainerStyle'])
  },
  methods: {
    // 使用 mapActions 将模块中的 actions 映射到方法中
    // 使用 mapActions 将 updateParentStyle action 映射为组件的方法
    ...mapActions('parentContainer', ['updateParentStyle']),

    // 方法：调用 Action 来修改父容器的样式
    changeStyle() {
      const newStyle = {
        backgroundColor: 'lightcoral',
        height: '350px'
      }
      this.updateParentStyle(newStyle) // 调用 Action 更新样式
    }
  }
}
</script>

