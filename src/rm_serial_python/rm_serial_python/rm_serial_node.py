import time
import serial
import threading
import struct
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header  # 字符串消息类型和头部消息类型
from rm_interfaces.msg import ArmorTracking , Decision # 导入自定义消息类型

class ColorPrint():
    def __init__(self):
        # 定义 ANSI 颜色代码
        self.PINK = "\033[38;5;218m"  # 亮紫色，接近粉色
        self.CYAN = "\033[96m"  # 亮青色
        self.GREEN = "\033[32m"  # 标准绿色
        self.RED = "\033[31m"  # 标准红色
        self.BLUE = "\033[34m"  # 标准蓝色
        self.RESET = "\033[0m"  # 重置颜色

class RMSerialDriver(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("启动 RMSerialDriver!")
        # 获取参数
        self.get_params()
        # 创建订阅者
        self.sub_tracker = self.create_subscription(
            ArmorTracking, "/tracker/target", self.send_data, 10
        )
        # 创建发布者
        self.pub_uart_receive = self.create_publisher(Decision, "/nav/decision", 10)
        # 创建变量
        self.tracking_color = 10
        self.timestamp = 0
        self.current_match = 0
        self.lagging = 0
        self.contect = 1

        self.color = ColorPrint()
        # 初始化串口
        try:
            self.serial_port = serial.Serial(
                port=self.device_name,
                baudrate=self.baud_rate,
                timeout=1,
                write_timeout=1,
            )
            if self.serial_port.is_open:
                self.get_logger().info("创建串口 successfully.")
                self.receive_thread = threading.Thread(target=self.receive_data)
                self.receive_thread.start()
                # self.timer = self.create_timer(1.0, self.receive_data_callback)  #有 bug 弃用
        except serial.SerialException as e:
            self.get_logger().error(f"创建串口时出错: {self.device_name} - {str(e)}")
            raise e

    def get_params(self):
        """获取并设置串口相关的参数"""
        self.device_name  = self.declare_parameter("device_name", "/dev/ttyUSB0").value
        self.baud_rate    = self.declare_parameter("baud_rate", 115200).value
        self.flow_control = self.declare_parameter("flow_control", "none").value
        self.parity       = self.declare_parameter("parity", "none").value
        self.stop_bits    = self.declare_parameter("stop_bits", "1").value

    def receive_data(self):
        """
        接收串口数据并处理
        
        预期格式:
        uint8_t header          - 1 byte
        uint8_t detect_color    - 1 byte
        float roll              - 4 bytes
        float pitch             - 4 bytes
        float yaw               - 4 bytes
        int32_t match           - 4 bytes
        uint16_t checksum       - 2 bytes
        总共: 1 + 1 + 4 + 4 + 4 + 4 + 2 = 8 bytes
        """
        
        serial_receive_msg = Decision()
        serial_receive_msg.header.frame_id = 'serial_receive_frame'
        serial_receive_msg.header = Header()
        serial_receive_msg.header.stamp = self.get_clock().now().to_msg()
        serial_receive_msg.color = 10  # 初始化颜色为 10

        # 计算接收的数据包长度
        packet_length = 20  # 1(header) + 1(color) + 4(roll) + 4(pitch) + 4(yaw) + 4(match) + 2(checksum)
        # 发布初始化的消息
        self.pub_uart_receive.publish(serial_receive_msg)
        self.get_logger().info("接收数据线程已启动")
        bytes_received = 0
        while rclpy.ok():
            try:
                # 更新消息头部
                serial_receive_msg.header.stamp = self.get_clock().now().to_msg()
                
                # 1. 查找帧头 (假设是0xA5，根据对面设备决定)
                header = None
                while header != b'\xA5' and rclpy.ok():
                    header = self.serial_port.read(1)
                    if not header:  # 超时
                        continue
                    bytes_received += 1
                    if header != b'\xA5':
                        self.get_logger().debug(f"跳过非头部字节: {header.hex()}")
                # 2. 头部找到，读取剩余数据包 (packet_length - 1 是因为我们已经读取了头部)
                remaining_data = self.serial_port.read(packet_length - 1)
                if len(remaining_data) != packet_length - 1:
                    self.get_logger().warn(f"数据包不完整: 预期 {packet_length-1} 字节, 实际收到 {len(remaining_data)} 字节")
                    continue
                bytes_received += len(remaining_data)
                # 3. 组合完整的数据包
                full_packet = header + remaining_data
                # 4. 分离数据和校验和
                data = full_packet[:-2]  # 除了最后2字节的校验和
                # 6. 校验和正确，解析数据
                # 从data中提取各个字段
                # struct格式: header(B), detect_color(B), roll(f), pitch(f), yaw(f), match(i)
                _, detect_color, roll, pitch, yaw, match = struct.unpack("<BBfffi", data)
                data = None
                self.get_logger().info(f"解包收到的数据: header=0x{header.hex()}, detect_color={detect_color}, "
                                    f"roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}, match={match:04d}")
                # 7. 更新ROS消息
                # 颜色字段
                if match != self.current_match:
                    if self.tracking_color != detect_color:
                        self.tracking_color = detect_color
                        serial_receive_msg.color = detect_color
                    self.contect = 1
                    
                elif self.contect == 1:
                    self.lagging += 1
                
                if self.lagging > 1000:
                    self.contect = 0
                    self.lagging = 0

                self.current_match = match
                # 8. 发布ROS消息
                self.pub_uart_receive.publish(serial_receive_msg)
                self.get_logger().info(f'{self.color.PINK}tracking_color:{self.color.RESET} {self.color.GREEN}{self.tracking_color}{self.color.RESET}')
                
            except (serial.SerialException, struct.error, ValueError) as e:
                self.get_logger().error(f"接收数据时出错: {str(e)}")
                self.reopen_port()
    def send_data(self, msg):
        """处理目标信息并通过串口发送"""
        try:
            # self.get_logger().info(f"发送数据: {msg}")
            header = 0x5A
            yaw    = msg.yaw
            pitch  = msg.pitch
            shoot = msg.shoot_flag

            self.timestamp += 1
            match = self.timestamp % 9999
            
            packet = struct.pack(
                "<BffBi", 
                header,
                yaw,
                pitch,
                shoot,
                match
            )
            self.serial_port.write(packet)
            # Verify packet length
            self.get_logger().info(f'{self.color.GREEN}数据:{self.color.RESET} {self.color.PINK}{packet}{self.color.RESET} ' 
                                    + f'{self.color.GREEN}长度:{self.color.RESET} {self.color.PINK}{len(packet)} bytes{self.color.RESET}'
                                    )
        except Exception as e:
            self.get_logger().error(f"发送数据时出错: {str(e)}")
            self.reopen_port()

    def reopen_port(self):
        """重新打开串口"""
        self.get_logger().warn("尝试重新打开串口")
        try:
            if self.serial_port.is_open:
                self.serial_port.close()
            self.serial_port.open()
            self.get_logger().info("成功重新打开串口")
        except Exception as e:
            self.get_logger().error(f"重新打开串口时出错: {str(e)}")
            time.sleep(1)
            self.reopen_port()

def main(args=None):  # ROS2节点主入口main函数
    rclpy.init(args=args)  # ROS2 Python接口初始化
    node = RMSerialDriver("rm_serial_python")  # 创建ROS2节点对象
    rclpy.spin(node)  # 循环等待ROS2退出
    node.destroy_node()  # 销毁节点对象
    rclpy.shutdown()  # 关闭ROS2 Python接口


if __name__ == "__main__":
    main()


