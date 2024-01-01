import torch
import ast
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


# 数据集类
class MovieDataset(Dataset):
    def __init__(self, data, image_folder="..\\data\\external\\img", transform=None) :
        self.data = data
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        movie_id = item['movieId']
        score = item['average']
        rating = item['rating']

        title_list = ast.literal_eval(item['title_vector'])
        title = torch.tensor(np.array(title_list)).float()

        tag_list = ast.literal_eval(item['tag_one_hot'])
        tag = torch.tensor(np.array(tag_list)).float()

        genres_list = ast.literal_eval(item['genres_one_hot'])
        genres = torch.tensor(np.array(genres_list)).float()

        features = torch.tensor(item.iloc[9:].values.astype(float)).float()

        # 将所有特征拼接成一个张量
        features = torch.cat((features, title, tag, genres), dim=0)

        # 加载图像或返回空白图像
        image_path = f"{self.image_folder}\\{movie_id}.png"

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            # 创建空白图像
            print(f"Warning: Image {image_path} not found. Creating a blank image.")
            image = Image.new('RGB', (176 ,176), (255, 255, 255))

        # 转换图像
        if self.transform:
            image = self.transform(image)
        
        image = torch.tensor(np.array(image)).float()

        return {
            'movie_id': movie_id,
            'score': score,
            'rating': rating,
            'features': features,
            'image': image
        }
