import os
import cv2
import json
import numpy as np

# 支持中文类标签
classes = ['silkworm','feed']  # 添加你的类别

# 基础路径（包含中文路径）
base_path = r'G:\chengjiayu\ultralytics-main\dataset\labelme'

# 列出文件名（不包含扩展名）
path_list = [os.path.splitext(i)[0] for i in os.listdir(base_path) if i.endswith('.json')]

# 遍历文件路径
for path in path_list:
    # 读取图像文件，处理中文路径
    image_path = os.path.join(base_path, f'{path}.jpg')
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 支持中文路径
    if image is None:
        print(f"无法读取图像文件: {image_path}")
        continue

    h, w, c = image.shape  # 获取图像尺寸

    # 打开 JSON 文件，确保编码为 UTF-8
    json_path = os.path.join(base_path, f'{path}.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        masks = json.load(f)['shapes']  # 解析 JSON 文件内容

    # 创建标签文件
    txt_path = os.path.join(base_path, f'{path}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        for idx, mask_data in enumerate(masks):
            mask_label = mask_data['label']
            if '_' in mask_label:
                mask_label = mask_label.split('_')[0]
            if mask_label not in classes:
                print(f"未找到类别: {mask_label}, 请在 'classes' 列表中添加该类别")
                continue
            mask = np.array([np.array(i) for i in mask_data['points']], dtype=np.float64)
            mask[:, 0] /= w  # 归一化 x 坐标
            mask[:, 1] /= h  # 归一化 y 坐标
            mask = mask.reshape((-1))
            if idx != 0:
                f.write('\n')
            f.write(f'{classes.index(mask_label)} {" ".join(map(lambda x: f"{x:.6f}", mask))}')
