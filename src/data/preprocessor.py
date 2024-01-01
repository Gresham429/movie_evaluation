import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path):
    """
    加载 csv 数据
    """
    return pd.read_csv(file_path)

def explore_data(df):
    """
    查看 csv 信息
    """
    print(df.head())
    print(df.info())
    print(df.describe())

def handle_missing_values(df):
    """
    处理缺失值
    """
    df.dropna(inplace=True)

def merge_tables(dataframes, on_column, how='outer'):
    """
    合并数据表
    """
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on=on_column, how=how)

    return merged_df

def handle_timestamp(df, column_name, time_unit='s'):
    """
    处理时间戳，将时间戳列转换为具体的年月日时分秒，并替换原始列。

    参数：
    - df: DataFrame，包含时间戳列的数据框。
    - column_name: str，包含时间戳的列名。
    - time_unit: str，时间戳的单位，默认为秒（'s'）。

    返回：
    - 无，结果替换了原始数据框中的时间戳列。
    """
    # 将时间戳列转换为 UTC 时间
    df[column_name] = pd.to_datetime(df[column_name], unit=time_unit, utc=True)

    # 根据需要选择其他时间格式
    df[column_name] = df[column_name].dt.strftime("%Y-%m-%d %H:%M:%S")

def get_unique_labels(df, column_name):
    """
    获取数据框中某列的所有唯一标签。

    参数：
    - df: DataFrame，包含目标列的数据框。
    - column_name: str，目标列的列名。

    返回：
    - labels: list，目标列的所有唯一标签。
    """
    labels = set()
    for column in df[column_name]:
        labels.update(column)

    return list(labels)

def one_hot_encode(df, column_name, unique_labels):
    """
    对数据框中的某列进行独热编码。

    参数：
    - df: DataFrame，包含目标列的数据框。
    - column_name: str，目标列的列名。
    - unique_labels: list，目标列的所有唯一标签。

    返回：
    - 无，结果替换了原始数据框中的目标列。
    """
    df[column_name + "_one_hot"] = df[column_name].apply(lambda x: [1 if label in x else 0 for label in unique_labels])

def preprocess_data(file_paths, external_file_path, on_column):
    """
    数据预处理主函数
    """
    # 加载数据
    dataframes = [load_data(file_path) for file_path in file_paths]

    # 处理缺失值
    for df in dataframes:
        handle_missing_values(df)

    # 合并数据表
    merged_df = merge_tables(dataframes, on_column)

    # 删除 userId_y 和 timestamp_y 列
    merged_df.drop(['userId_y', 'timestamp_y'], axis=1, inplace=True)

    # 处理时间戳
    handle_timestamp(merged_df, 'timestamp_x')

    # 分词并且转换为小写
    merged_df['genres'] = merged_df['genres'].apply(lambda x: x.lower().split('|'))

    # 将 NaN 替换为 'unknown'
    merged_df['tag'].fillna('unknown', inplace=True)

    def merge_and_unique_lists(lst):
        merged_list = [item for item in lst]
        return list(set(merged_list))

    # 先按照 movieId 分组，然后对每个组内的 tag 列执行合并和去重操作
    merged_df = merged_df.groupby('movieId').agg({'title': 'first', 'genres': 'first', 'tag': merge_and_unique_lists, 'rating': 'mean'}).reset_index()

    # 评级映射
    bins = [0, 0.5, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # 删除 'rating' 列中包含空缺值的行
    merged_df = merged_df.dropna(subset=['rating'])
    
    # 重命名 'rating' 列为 'average'
    merged_df.rename(columns={'rating': 'average'}, inplace=True)

    merged_df['rating'] = pd.cut(merged_df['average'], bins=bins, labels=labels, include_lowest=True)

    # 提取"title"列数据
    titles = merged_df['title'].tolist()

    # 预处理和分词
    tokenized_titles = [word_tokenize(title.lower()) for title in titles]

    # 创建Word2Vec模型
    title_model = Word2Vec(tokenized_titles, vector_size=100, workers=4, window=5, min_count=1, sg=1)

    # 定义函数以获取文本的向量表示
    def get_title_vector(title):
        words = title.lower().split()
        vectors = [title_model.wv[word] for word in words if word in title_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100, dtype=np.float32)
    
    # 对"title"列应用函数并创建新的向量列
    merged_df['title_vector'] = merged_df['title'].apply(get_title_vector)

    # 将向量表示转换为字符串形式
    def vector_to_string(vector):
        return ','.join(str(num) for num in vector)

    merged_df['title_vector'] = merged_df['title_vector'].apply(vector_to_string)

    # 将 'genres' 列转换为独热编码
    unique_genres = get_unique_labels(merged_df, 'genres')
    one_hot_encode(merged_df, 'genres', unique_genres)

    # 将 'tag' 列转换为独热编码
    unique_tags = get_unique_labels(merged_df, 'tag')
    one_hot_encode(merged_df, 'tag', unique_tags)

    # 加载爬下来的数据并合并处理
    external_df = load_data(external_file_path)

    merged_df = merge_tables([merged_df, external_df], on_column, how='left')

    merged_df.iloc[:, 6:] = merged_df.iloc[:, 6:].fillna(0)

    # 指定需要标准化的列
    columns_to_normalize = ["reviews", "critic_reviews", "metascore", "Budget", "US_and_Canada_Gross", "Opening_Weekend", "Global_Gross", "Runtime"]

    # 创建标准化器
    scaler = MinMaxScaler()

    # 对指定列进行标准化处理
    merged_df[columns_to_normalize] = scaler.fit_transform(merged_df[columns_to_normalize])

    # # 查看信息
    # explore_data(merged_df)

    return merged_df
