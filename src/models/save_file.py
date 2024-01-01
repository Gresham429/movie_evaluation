import os
import torch


def find_save_path(model_name, save_path, file_suffix):
    # 获取目标文件夹内所有文件的列表
    file_list = os.listdir(save_path)

    # 找出最大的数字后缀
    max_suffix = 0
    for file in file_list:
        if file.startswith(f'{model_name}_') and file.endswith(file_suffix):
            suffix = int(file.split('_')[-1].split('.')[0])
            max_suffix = max(max_suffix, suffix)

    # 构建新的模型名称
    new_suffix = max_suffix + 1
    new_model_name = f'{model_name}_{new_suffix}' + file_suffix

    # 添加后缀，在新的模型名称中检查是否已存在同名文件
    while new_model_name in file_list:
        new_suffix += 1
        new_model_name = f'{model_name}_{new_suffix}' + file_suffix

    # 更新模型保存路径
    new_save_path = os.path.join(save_path, new_model_name)

    return new_save_path