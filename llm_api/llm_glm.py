import httpx
import re
import json



def ask_glm(json_data):
    timeout = httpx.Timeout(timeout=100.0)
    url = "http://127.0.0.1:9000/chat/chat" 
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=json_data)
            response.raise_for_status() 
            parsed_data = json.loads(response.text)
            return parsed_data['response']
    except (httpx.RequestError,ValueError): 
        print("请求ChatGlm2-6b大模型出错")


def chat(prompt):
    
    length = len(prompt) +100
    json_data = {
        "query": prompt,
        # "history":'',
        "top_p": 0.9,
        "temperature": 0.5,
        "max_length": length,
        "stream": False,
        "model_name": "chatglm2-6b"
    }
    response = ask_glm(json_data)
    return response
if __name__=='__main__':
    answer = chat('你好')
    print(answer)
    