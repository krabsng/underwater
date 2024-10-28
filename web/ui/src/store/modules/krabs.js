// 定义应用程序的状态数据
const state = {
  parentContainerStyle: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center'
  }
}

// 修改状态数据的同步方法
const mutations = {
  UPDATE_PARENT_STYLE(state, newStyle) {
    state.parentContainerStyle = { ...state.parentContainerStyle, ...newStyle }
  }
}

// 可以包含异步操作的动作，例如 API 请求
const actions = {
  updateParentStyle({ commit }, newStyle) {
    commit('UPDATE_PARENT_STYLE', newStyle)
  }
}

export default {
  namespaced: true,
  state,
  mutations,
  actions
}
