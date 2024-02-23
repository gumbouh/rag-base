prompt_dict = {
    'judge':'请帮我判断问题与检索内容是否相关,回答True或者False,问题:{question},检索内容:{context}\n回答:',
    'retrieval': '问题:{question}\n跟问题相关的文档:{context}\n请根据文档和问题给出相应的回答,回答需引用文档原文，并作出一定解释\n回答:',
    
}
def get_prompt(type, args):
    prompt = ''
    if type == 'retrieval':
        prompt = prompt_dict[type].format(question=args.question, context=args.context)
    elif type == 'judge':
        prompt = prompt_dict[type].format(question=args.question, context=args.context)
    return prompt
