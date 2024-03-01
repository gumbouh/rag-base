import httpx
import re
import json



def ask_llama(json_data):
    # print(json_data)
    timeout = httpx.Timeout(timeout=100.0)
    url = "http://127.0.0.1:8000/v1/chat/completions" 
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=json_data)
            response.raise_for_status() 
            # print(response)
            parsed_data = json.loads(response.text)
            print(parsed_data)
            return parsed_data['choices'][0]['message']['content']
    except (httpx.RequestError,ValueError): 
        print("请求Llama-api的大模型出错")


def chat(prompt):
    
    length = len(prompt) +100
    json_data = {
        "model": "string",
        "messages": [
            {
            "role": "user",
            "content": prompt,
            }
        ],
        "tools": [],
        "do_sample": True,
        "temperature": 0.95,
        "top_p": 0.7,
        "n": 1,
        "max_tokens": length,
        "stream": False}    
    response = ask_llama(json_data)
    return response
if __name__=='__main__':
    answer = chat('你好')
    print(answer)
    