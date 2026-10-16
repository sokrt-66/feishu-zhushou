import requests
import time

time.sleep(3)

# 1. 获取 admin 的 Token
response = requests.post("http://localhost:5001/token", data={"username": "admin", "password": "password"})
print("Token获取:", response.status_code)
token = response.json().get("access_token")
headers = {"Authorization": f"Bearer {token}"}

# 2. 添加员工小雪
employee_data = {
    "name": "小雪",
    "gender": "女",
    "department": "技术部",
    "position": "后端开发助手",
    "hire_date": "2025-05-03",
    "salary": 4500,
    "email": "1234567891@qq.com",
    "phone": "12345678971",
    "status": "在职"
}

response = requests.post("http://localhost:5001/api/employees", headers=headers, json=employee_data)
print("添加员工结果状态:", response.status_code)
print("添加员工结果文本:", response.text)

# 3. 查询技术部员工列表
response = requests.get("http://localhost:5001/api/employees", headers=headers)
print("\n员工列表状态:", response.status_code)
print("员工列表文本:", response.text)