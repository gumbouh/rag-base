import openai
import time
import os
os.environ['OPENAI_API_KEY'] = 'sk-vAgCvOV8pFu0YvOj8GjGirY6LAOyWFYyZ26EubXSU48Wa1Zx'
# os.environ['OPENAI_API_KEY'] = 'sk-efMVlpOpSO2nV3if6mOI5rzLSxX6dhlVVUBZ2o6E3hvC5aHX'
openai.api_base = 'https://api.chatanywhere.com.cn/v1'
# openai.api_key = 'sk-efMVlpOpSO2nV3if6mOI5rzLSxX6dhlVVUBZ2o6E3hvC5aHX'
# os.environ["HTTP_PROXY"] = "127.0.0.1:20171"
# os.environ["HTTPS_PROXY"] = "127.0.0.1:20171"
from langchain.chat_models import ChatOpenAI

def chat(prompt):
    while True:
        try:
            gpt = ChatOpenAI(model_name='gpt-3.5-turbo',openai_api_key=os.getenv("OPENAI_API_KEY"),openai_api_base='https://api.chatanywhere.com.cn/v1')
            answer = gpt.predict(prompt)
            break
            
        except Exception as e:
            print(f"error{e}")
            continue
    return answer

if __name__=='__main__':
    answer = chat('你好')
    print(answer)
    