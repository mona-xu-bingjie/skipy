from openni import openni2
import numpy as np
import cv2
import random
import torch
import rclpy
from rclpy.node import Node
from ultralytics import YOLO



from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D

model= torch.hub.load('ultralytics/yolov5', 'yolov5m')
#model = YOLO("yolov8n.pt")
model.conf = 0.45

depth_f=[581.315,581.315] #ir相机的内参_fx,fy
depth_c=[316.116,243.899] #ir相机的内参_u0,v0
rgb_f=[543.415,543.415]#rgb相机的内参_fx,fy
rgb_c=[320.017,247.56]#rgb相机的内参_u0,v0


def depth_pixel2cam(u, v, depth, fx, fy, cx, cy):
        # 将像素坐标(u, v)处的深度值depth转换为相机坐标系中的点
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return x, y, z


def get_mid_pos(frame,box,depth_data,randnum):
    distance_list = []
    mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist.any():
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
    #print(distance_list, np.mean(distance_list))
    return np.mean(distance_list)

 

class MPublisher(Node):

    def __init__(self):
        super().__init__('publisher')
        self.publisher_ = self.create_publisher(Detection2DArray,"publisher",10)
        self.result_msg = Detection2DArray()
        openni2.initialize()
        dev = openni2.Device.open_any()
        print(dev.get_device_info())
        depth_stream = dev.create_depth_stream()
        dev.set_image_registration_mode(True)
        depth_stream.start()
        cap = cv2.VideoCapture(2)
        cv2.namedWindow('depth')
        while True:
    
            frame = depth_stream.read_frame()
            #转换数据格式
            dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])
            dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
            dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
    
            dpt2 *= 255
            dpt = dpt1 + dpt2
            #cv2里面的函数，就是类似于一种筛选
            #假设需要让深度摄像头感兴趣的距离范围有差别地显示，那么我们就需要确定一个合适的alpha值，公式为：有效距离*alpha=255
            #假设我们想让深度摄像头8m距离内的深度被显示，>8m的与8m的颜色显示相同，那么alpha=255/(8*10^3)≈0.03
            #假设我们想让深度摄像头6m距离内的深度被显示，>6m的与6m的颜色显示相同，那么alpha=255/(6*10^3)≈0.0425
            dim_gray = cv2.convertScaleAbs(dpt, alpha=0.03)
            #对深度图像进行一种图像的渲染，目前有11种渲染方式
            depth_colormap = cv2.applyColorMap(dim_gray, 2)  # 有0~11种渲染的模式
            flipped_image = cv2.flip(depth_colormap, 1) 
            cv2.imshow('depth', flipped_image)

    
            ret, frame = cap.read()
            results = model(frame)
            boxs= results.pandas().xyxy[0].values

            #self.get_logger().info(str(results))
            #self.result_msg.detections.clear()
            #self.result_msg.header.frame_id = "camera"
            #self.result_msg.header.stamp = self.get_clock().now().to_msg()
            
            #MPublisher.dectshow(self,frame, boxs, depth_colormap)
            #cv2.imshow('color', frame)
            img = frame.copy()
            height, width = 480,640
            cam_coord = np.zeros((height, width, 3))  # 创建一个数组来存储相机坐标系中的点
            for box in boxs:

                detection2d = Detection2D()
                detection2d.id = str(box[-1])
                #print(box)

                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                dist = get_mid_pos(frame, box, depth_colormap, 100)
                #print(dist/50)
                mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2]
                    
                depth = depth_colormap[np.uint8(mid_pos[1]), np.uint8(mid_pos[0])]  # 获取深度图中像素(u, v)处的深度值
                #print(depth)
                cam_coord= depth_pixel2cam(mid_pos[0], mid_pos[1], dist, depth_f[0], depth_f[1], depth_c[0], depth_c[1])
                #print(cam_coord)
                #cv2.putText(img,box[-1]+ " x:"+ str(cam_coord[0]//1)+ " y:" +str(cam_coord[1]//1)+" z:" +str(dist//1),
                            #(int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(img,box[-1]+ " dis:"+ str(dist/100)[:4] + 'm',
                            (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                #print(box[-1]+" x:" + str(cam_coord[0][0]//1) + " y:" +str(cam_coord[1][0]//1)+" z:" +str(cam_coord[2][0]//1)+ " dis:"+ str(dist / 1000)[:4] + 'm')
                
                detection2d.bbox.center.position.x = float(mid_pos[0])
                detection2d.bbox.center.position.y = float(mid_pos[1])

                obj_pose = ObjectHypothesisWithPose()

                obj_pose.hypothesis.class_id = str(box[-1])
                obj_pose.hypothesis.score=float(box[-2]//100)

                obj_pose.pose.pose.position.x = float(cam_coord[0]//1)
                obj_pose.pose.pose.position.y = float(cam_coord[1]//1)
                obj_pose.pose.pose.position.z = float(dist//1)
                detection2d.results.append(obj_pose)
                self.result_msg.detections.append(detection2d)
            self.publisher_.publish(self.result_msg)
            #self.get_logger().info('Publishing: "%s"' % self.result_msg)

            cv2.imshow('dec_img', img)
            key = cv2.waitKey(1)
            if int(key) == ord('q'):
                 break   
            
        depth_stream.stop()
        dev.close()

def main():
    rclpy.init()
    rclpy.spin(MPublisher())
    rclpy.shutdown()

if __name__ == "__main__":
    main()