# 测试限流（70 个请求，需要 Python + requests）
import requests
import time

# 获取 token
token_resp = requests.post(
    "http://localhost:5000/token",
    data={"username": "admin", "password": "password"}
)
token = token_resp.json()["access_token"]

url = "http://localhost:5000/api/message"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
payload = {"sender_id": "t", "content": "test", "message_type": "text"}

for i in range(1, 71):
    resp = requests.post(url, json=payload, headers=headers)
    print(f"{i:2d} -> {resp.status_code}")
    if resp.status_code == 429:
        print("限流触发！")
        break
    time.sleep(0.05)