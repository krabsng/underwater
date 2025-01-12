import request from '@/utils/request'

export function login(data) {
  return request({
    url: 'login/',
    method: 'post',
    data
  })
}

export function userRegister(data) {
  return request({
    url: 'register/',
    method: 'post',
    data
  })
}

export function getInfo(token) {
  return request({
    url: '/user/',
    method: 'get',
    params: { token }
  })
}

export function logout() {
  return request({
    url: '/vue-admin-template/user/logout',
    method: 'post'
  })
}
