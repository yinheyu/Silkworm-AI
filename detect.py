import warnings

import pandas as pd
import torch

warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 推理参数官方详解链接：https://docs.ultralytics.com/modes/predict/#inference-sources:~:text=of%20Results%20objects-,Inference%20Arguments,-model.predict()

if __name__ == '__main__':
    model = YOLO(r'G:\chengjiayu\ultralytics-main\runs\train\lsk+GLSA+ContextGuidedDown\weights/best.pt') # select your model.pt path
    # model.predict(source=r'G:\chengjiayu\fanqie\ultralytics-main\dataset\fanqieshipin\DJI_20241017151202_0001_D.MP4',
    results = model.track(source=r'G:\chengjiayu\ultralytics-main\dataset\custom_dataset蚕\images\val' ,
                  imgsz=(640,640),
                  project='runs/detect',
                  name='exp',
                  save=True,
                  conf=0.2,
                  # iou=0.35,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  line_width=1, # line width of the bounding boxes
                  show_conf=False, # do not show prediction confidence
                  show_labels=False, # do not show prediction labels
                  save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )
    # for r in results:
    #      print(r.boxes.id)  # print tracking IDs

    # 选择特定帧的结果
    # frame_results = results[0]  # 例如，选择第一帧的结果

    # # 获取检测框信息
    # boxes = frame_results.boxes
    #
    # # 假设我们要获取ID为3的检测框坐标
    # target_id = 3
    #
    # # 找到ID为3的检测框索引
    # indices = torch.where(boxes.id == target_id)[0]
    #
    # # 获取对应ID的坐标
    # if indices.numel() > 0:
    #     coordinates = boxes.xyxy[indices]
    #     print(f"Coordinates for ID {target_id}: {coordinates}")
    # else:
    #     print(f"No object found with ID {target_id}")
    # 处理每一帧的结果
    # 初始化目标数据列表，用于保存每一帧的目标信息
    # columns = ['Frame', 'Target ID', 'Class', 'x1', 'y1', 'x2', 'y2']
    # data = []
    #
    # previous_boxes = {}  # 用于存储上一帧的目标框信息（目标ID -> [x1, y1, x2, y2]）
    # current_id = 0  # 唯一ID初始化
    #
    # # 处理每一帧的结果
    # for frame_idx, frame_results in enumerate(results):
    #     # 获取当前帧的检测框
    #     boxes = frame_results.boxes
    #
    #     # 提取类别信息和坐标
    #     classes = boxes.cls.cpu().numpy()  # 获取类别
    #     xyxy = boxes.xyxy.cpu().numpy()  # 获取坐标 (x1, y1, x2, y2)
    #     ids = boxes.id.cpu().numpy()  # 获取目标ID
    #
    #     # 提取类别0和类别1的目标框
    #     class_0_indices = [i for i, cls in enumerate(classes) if cls == 0]  # 类别0
    #     class_1_indices = [i for i, cls in enumerate(classes) if cls == 1]  # 类别1
    #
    #     class_0_boxes = [xyxy[i] for i in class_0_indices]
    #     class_1_boxes = [xyxy[i] for i in class_1_indices]
    #
    #     # 按x坐标（从左到右）排序
    #     class_0_boxes_sorted = sorted(zip(class_0_boxes, ids), key=lambda x: x[0][0])  # 排序依据是x1
    #     class_1_boxes_sorted = sorted(zip(class_1_boxes, ids), key=lambda x: x[0][0])  # 排序依据是x1
    #
    #     # 记录当前帧的目标ID
    #     current_boxes = {}
    #
    #     # 为类别0和类别1的目标分配ID，并去除重复目标
    #     for box, target_id in class_0_boxes_sorted:
    #         current_boxes[target_id] = box
    #     for box, target_id in class_1_boxes_sorted:
    #         current_boxes[target_id] = box
    #
    #     # 更新上一帧的目标框信息
    #     previous_boxes = current_boxes
    #
    #     # 将当前帧的目标数据添加到 `data` 列表
    #     for target_id, box in current_boxes.items():
    #         cls = classes[ids.tolist().index(target_id)]  # 通过目标ID从classes中获取目标类别
    #         data.append([frame_idx, target_id, cls, box[0], box[1], box[2], box[3]])
    #
    #     # 打印当前帧的目标ID及其坐标
    #     print(f"Frame {frame_idx}:")  # 使用 `enumerate()` 来跟踪帧的索引
    #     for target_id, box in current_boxes.items():
    #         print(f"ID: {target_id}, Box: {box}")
    #
    # # 将数据保存到 Excel 文件
    # df = pd.DataFrame(data, columns=columns)
    # df.to_excel(r"G:\chengjiayu\fanqie\ultralytics-main/tracking_results.xlsx", index=False, engine='openpyxl')  # 保存为 Excel 文件
    # print("Results saved to 'tracking_results.xlsx'")








