# 
# 检索增强工具类
# 
def get_retriver(vector_path, model_path):
    from langchain.embeddings import HuggingFaceBgeEmbeddings
    from langchain.vectorstores import Chroma
    model_kwargs = {'device': 'cuda:0'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceBgeEmbeddings(
                    model_name=model_path,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    query_instruction="为文本生成向量表示用于文本检索"
                )
    # 
    vectordb = Chroma(persist_directory=vector_path, embedding_function=embedding)
    return vectordb

def get_reranker(reranker_path):
    from FlagEmbedding import FlagReranker
    reranker = FlagReranker(reranker_path, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    return reranker

def clean_context(context):
    content = []
    for idx, c in enumerate(context):
        content_str = "文档信息:"+str(c[0].metadata)+'\n文档内容:'+c[0].page_content+"\n "
        
        content.append(content_str)
    return content
def get_clean_context(context):
    # content = []
    content_str = ""
    for idx, c in enumerate(context):
        # print(c[0].page_content)
        # content.append(c[0].page_content)
        content_str +="文档["+str(idx)+"]: "
        content_str += str(c)+"\n "
    return content_str
def construct_pair(doc, context):
    pair = []
    for c in context:
        pair.append([doc, c])
    return pair

def rerank(reranker,doc,context,topk=5):

    context = clean_context(context)
    # print('context:{}'.format(context))
    pair = construct_pair(doc, context)
    # print('pair:{}'.format(pair))


    scores = reranker.compute_score(pair)
    # print('scores:{}'.format(scores))
    # 根据scores相似度分数，给context重新排序 
    # 将context和scores组合在一起
    context_scores = list(zip(context, scores))
    # 根据scores进行排序
    context_scores_sorted = sorted(context_scores, key=lambda x: x[1], reverse=True)
    # 提取排序后的context
    context_sorted = [cs[0] for cs in context_scores_sorted[:topk]]
    # print('context_sorted:{}'.format(context_sorted))
    # format
    
    return context_sorted
