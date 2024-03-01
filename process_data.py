import pandas as pd
def process_data(path):
    df = pd.read_csv(path)
    df = df[['sentence_sep', 'label_raw']]
    df.to_csv('data/cmcc/konwledgebase.csv', index=False)

def create_konwledge_base(path):
    # 读取CSV文件
    df = pd.read_csv(path)

    # 获取所有的标签类别
    labels = df['label_raw'].unique()

    # 初始化一个空的DataFrame来存储选取的样本
    new_df = pd.DataFrame()

    # 对于每个标签类别，随机选择30个样本（如果存在的话）
    for label in labels:
        samples = df[df['label_raw'] == label].sample(n=30, replace=True)
        new_df = pd.concat([new_df, samples])

    # 保存新的DataFrame为小的dev.csv
    new_df.to_csv('data/cmcc/konwledgebase.csv', index=False)
    
if __name__=='__main__':
    # path ='/home/gumbou/codespace/abst-relate/HAN/data/all/dev.csv'
    # create_konwledge_base(path)
    process_data('data/cmcc/konwledgebase.csv')