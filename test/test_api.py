#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试所有API端点"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_api(endpoint, method="GET", headers=None, body=None, expected_status=200):
    """测试单个API端点"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=body)
        
        status_ok = response.status_code == expected_status
        if status_ok:
            print(f"✅ [{method}] {endpoint} - {response.status_code}")
            if response.content:
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        keys = list(data.keys())[:5]
                        print(f"   响应字段: {', '.join(keys)}")
                except:
                    print(f"   响应内容: {response.text[:100]}...")
        else:
            print(f"❌ [{method}] {endpoint} - {response.status_code}")
            print(f"   错误信息: {response.text[:200]}")
        
        return status_ok
    except Exception as e:
        print(f"❌ [{method}] {endpoint} - 连接失败: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("开始测试API端点...")
    print("=" * 60)
    
    # 1. 健康检查
    test_api("/health")
    
    # 2. 获取token
    print("\n--- 获取访问令牌 ---")
    token_response = requests.post(f"{BASE_URL}/token", 
                                  data={"username": "admin", "password": "password"},
                                  headers={"Content-Type": "application/x-www-form-urlencoded"})
    
    if token_response.status_code == 200:
        token = token_response.json().get("access_token")
        print(f"✅ 获取令牌成功")
        auth_headers = {"Authorization": f"Bearer {token}"}
    else:
        print(f"❌ 获取令牌失败: {token_response.status_code}")
        auth_headers = None
    
    # 3. 测试需要认证的API
    print("\n--- 测试需要认证的API ---")
    
    if auth_headers:
        test_api("/api/tasks/list", headers=auth_headers)
        test_api("/api/permissions/kbs", headers=auth_headers)
        test_api("/api/permissions/department", headers=auth_headers)
        test_api("/api/meeting/history", headers=auth_headers)
        test_api("/api/data/query", method="POST", headers=auth_headers, body={"query": "查询测试"})
        
        # 测试创建任务
        test_api("/api/tasks/create", method="POST", headers=auth_headers, 
                 body={"title": "测试任务", "description": "测试描述"})
        
        # 测试创建知识库
        test_api("/api/permissions/kb/create", method="POST", headers=auth_headers,
                 body={"kb_id": "test_kb", "name": "测试知识库"})
    
    print("\n" + "=" * 60)
    print("API测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
