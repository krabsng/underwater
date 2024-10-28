from datetime import datetime

# 返回json
def create_response(status, code, message, data=None):
    return {
        "status": status,                  # 响应状态
        "code": code,                      # 状态码
        "message": message,                # 反馈信息
        "data": data,                      # 具体数据内容
        "timestamp": datetime.utcnow().isoformat()  # 响应时间戳
    }
