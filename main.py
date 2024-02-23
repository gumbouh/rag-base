from rag_utils import get_retriver,get_clean_context,rerank,get_reranker
from prompt.wenshi_prompt import get_prompt
from llm_api.llm_gpt import chat
import argparse


VECTOR_PATH='vector/wenshi/db-2'
EMBEDDING_MODEL_PATH='/home/gumbou/models/m3e-base'
RERANKER_PATH = '/home/gumbou/models/bge-reranker-base'
TOPK_retrieve = 10
TOPK_rerank = 5
QUESTIONS =['有一名候选人已经在六家上司公司担任董事了，他还能担任我们温氏集团的董事吧','董事会秘书聘任需要提供什么材料','作为温氏集团的子公司，要提供哪些报表','股东要培训吗','公司内部控制自我评价报告包含什么内容','公司在新闻发布会上传达信息要提前审批吗']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    vectordb =get_retriver(VECTOR_PATH, EMBEDDING_MODEL_PATH)
    # 获取问题question
    question = QUESTIONS[0]
    # 检索
    context = vectordb.similarity_search_with_score(question, TOPK_retrieve)
    # clean_context = get_clean_context(context)
    # print('context:{}'.format(context))
    reranker = get_reranker(RERANKER_PATH)
    # 重排
    rerank_context = rerank(reranker,question,context,TOPK_rerank)
    # print('rerank:{}'.format(rerank_context))
    # clean_context = get_clean_context(rerank_context)
    # print('rerank:{}'.format(clean_context))
    args.context = context
    args.question = question
    
    # # 判断问题与检索内容是否相关
    # type = 'judge'
    # judge_prompt = get_prompt(type, args)
    # ans = chat(judge_prompt)
    # print(ans)
    
    # # 不相关让用户补充问题
    # question2= ''
    
    # 生成答案
    type = 'retrieval'
    retrieval_prompt = get_prompt(type, args)
    ans  = chat(retrieval_prompt)
    print(ans)