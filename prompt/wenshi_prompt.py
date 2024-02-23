prompt_dict = {
    'retrieval': '问题:{question}\n跟问题相关的文档:{docs}\n请根据文档和问题给出相应的回答:',
}
def get_prompt(type, args):
    prompt = ''
    if type == 'retrieval':
        prompt = prompt_dict[type].format(question=args.question, docs=args.docs)
    
    return prompt
