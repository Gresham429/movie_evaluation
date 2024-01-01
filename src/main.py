import sys
import nltk

# 检查标志变量，如果尚未下载资源，则下载
if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt')

import warnings
warnings.filterwarnings("ignore")

from numpy import average
from config.read_config import load_config
from data.preprocessor import preprocess_data, load_data
from data.crawler import crawl_links
from data.crawler_img import crawl_img
from models.mlp import mlp_predict
from models.cnn_1D import cnn_1d_predict
from models.cnn_2D import cnn_2d_predict
from models.random_forest import random_forest_predict


if __name__ == "__main__":
    # 读取配置文件
    config = load_config()
    preprocess_config = config['preprocess']
    file_paths = [preprocess_config['paths'][f'file_path{i}'] for i in range(1, 4)]
    external_file_path = preprocess_config['paths']['file_path4']
    on_column = preprocess_config['options']['on_column']
    output_file = preprocess_config['paths']['output_file']
    links_file = preprocess_config['paths']['links_file']
    external_folder_path = preprocess_config['paths']['external_folder_path']

    # 判断命令行参数
    if "--crawl-info" in sys.argv:
        # 运行爬虫主函数
        print("正在爬取数据：")
        crawl_links(links_file, external_file_path)

        print("数据爬取完成。")

    elif "--crawl-img" in sys.argv:
        # 运行爬虫主函数
        print("正在爬取图片：")
        crawl_img(links_file, external_folder_path)

        print("图片爬取完成。")

    elif "--preprocess" in sys.argv:
        # 运行数据预处理主函数
        print("正在处理数据：")
        preprocessed_data = preprocess_data(file_paths, external_file_path, on_column)

        # 保存处理后的数据
        preprocessed_data.to_csv(output_file, index=False)

        print("数据处理完成。")

    elif "--train-and-predict" in sys.argv:
        # 评级预测
        print("正在训练模型并预测评级：")
        preprocessed_data = load_data(output_file)  # 加载预处理后的数据

        accuracy = []

        if "--mlp" in sys.argv:          
            print("MLP模型：")

            for idx in range(3):
                print("第", idx + 1, "次训练：")

                score = mlp_predict(preprocessed_data)
                accuracy.append(score)

            print("三次训练的准确率分别为：", accuracy)

            average_accuracy = average(accuracy)

            print("Average MLP Accuracy:", average_accuracy)

        elif "--cnn" in sys.argv:     
            print("一维 CNN 模型：")

            for idx in range(3):
                print("第", idx + 1, "次训练：")

                score = cnn_1d_predict(preprocessed_data)
                accuracy.append(score)

            print("三次训练的准确率分别为：", accuracy)
            
            average_accuracy = average(accuracy)

            print("Average CNN Accuracy:", average_accuracy)

        elif "--cnn-2d" in sys.argv:     
            print("二维 CNN 模型：")

            for idx in range(3):
                print("第", idx + 1, "次训练：")

                score = cnn_2d_predict(preprocessed_data)
                accuracy.append(score)

            print("三次训练的准确率分别为：", accuracy)
            
            average_accuracy = average(accuracy)

            print("Average CNN 2D Accuracy:", average_accuracy)

        elif "--random-forest" in sys.argv:
            print("Random Forest模型：")

            for idx in range(3):
                print("第", idx + 1, "次训练：")

                score = random_forest_predict(preprocessed_data)
                accuracy.append(score)

            average_accuracy = average(accuracy)

            print("三次训练的准确率分别为：", accuracy)

            print("Average Random Forest Accuracy:", average_accuracy)

