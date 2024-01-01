# 电影评分预测

### 硬件前提

有 NVIDIA 显卡并且安装 cuda ， 可用显存 > 5 GB。（ CPU 上跑不动 CNN ，只能跑随机森林 ）

### 环境配置

##### conda

```shell
cd path/to/root
conda create --name movie_evaluation python=3.10.13
conda activate movie_evaluation
pip install -r requirements.txt
```

##### 项目

请按照项目结构说明，在 data 中新建 processed 文件夹，在根目录下新建 weight 文件夹。

### 运行测试

1、参考 config.json，配置 config_default.json。

2、爬虫测试。

```shell
cd path/to/root/src
# 爬取数据文件，如电影时长、评论数等，并存入csv
python main.py --crawl-info
# 爬取电影海报
python main.py --crawl-img
```

3、数据清洗与预处理

```shell
python main.py --preprocess
```

4、模型预测分析

```shell
# Random Forest
python main.py --train-and-predict --random-forest

# MLP 多层感知机
python main.py --train-and-predict --mlp

# CNN 卷积神经网络
python main.py --train-and-predict --cnn

# CNN 二维卷积神经网络
python main.py --train-and-predict --cnn-2d
```

### 项目结构

```shell
├───data
│   ├───external
│   │   └───img
│   ├───processed
│   └───raw
├───notebooks
├───src
│   ├───config
│   ├───data
│   └───models
└───weight
```

1. `data`：该文件夹包含与项目相关的数据。
   - `external`：包含外部数据，包括`img`子文件夹中的图像文件。
   - `processed`：包含处理过的数据。
   - `raw`：包含原始未经处理的数据。
2. `notebooks`：存放训练的 loss、accuracy 等曲线。
3. `src`：该文件夹包含项目的源代码。
   - `config`：包含配置文件读取代码。
   - `data`：包含与数据相关的代码，包括爬虫、数据预处理、dataset定义等部分。
   - `models`：包含与模型相关的代码，包括 Random Forest 、MLP 、CNN 等模型定义以及权重文件的保存 。
4. `weight`：包含已训练模型的权重文件。