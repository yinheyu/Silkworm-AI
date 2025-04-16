import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

if __name__ == '__main__':
    model = YOLO(r'G:\chengjiayu\ultralytics-main\runs\train\v9c梨花\weights/best.pt')
    model.val(data=r'G:\chengjiayu\ultralytics-main\dataset/data.yaml',
              split='val',
              imgsz=640,
              batch=1,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/test',
              name='exp',
              )