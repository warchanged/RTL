import requests
import json

def vllm_generate(prompt, max_tokens=1024):
    url = "http://127.0.0.1:8000/v1/completions"
    data = {
        "model": "results/my_rlt_model/checkpoint-50",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    response = requests.post(url, json=data)
    return response.json()["choices"][0]["text"]

# 测试
prompt = "请分析这个数学问题的解决步骤：2x-3=5, 求 x= ?"
response = vllm_generate(prompt)
print(response)