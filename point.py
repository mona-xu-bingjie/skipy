import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import time

class PclListener(Node):

    def __init__(self):
        super().__init__('pcl_listener')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT,
            history=QoSHistoryPolicy.SYSTEM_DEFAULT,
            depth=5
        )
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.callback_pointcloud,
            qos_profile
        )
        
    def callback_pointcloud(self, data):
        assert isinstance(data, PointCloud2)
        gen = point_cloud2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
        time.sleep(1)
        self.get_logger().info('PointCloud2:')
        for p in gen:
            self.get_logger().info(" x : %.3f  y: %.3f  z: %.3f" %(p[0],p[1],p[2]))


def main(args=None):
    rclpy.init(args=args)
    pcl_listener = PclListener()
    rclpy.spin(pcl_listener)
    pcl_listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
