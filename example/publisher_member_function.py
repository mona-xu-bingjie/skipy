# 训练模型
from ultralytics import YOLO
 
model = YOLO("yolov8n.yaml")
model.train(data="coco128.yaml", epochs=5) #详细参数见2.2
 
# 验证模型
model.val()
 
# 预测模型
model.predict(source="2") # 0是摄像头，详细参数见3.2
 
# 按指定格式导出模型
model.export(format="onnx")