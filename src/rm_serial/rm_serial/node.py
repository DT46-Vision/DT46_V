import time
import serial
import threading
import rclpy
from rclpy.node import Node
from rm_interfaces.msg import Decision, GimbalControl
import struct
from geometry_msgs.msg import Vector3Stamped
from .modules.crc import *
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data

class ColorPrint():
    def __init__(self):
        self.PINK = "\033[38;5;218m"
        self.CYAN = "\033[96m"
        self.GREEN = "\033[32m"
        self.RED = "\033[31m"
        self.BLUE = "\033[34m"
        self.RESET = "\033[0m"

class SerialNode(Node):
    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info("启动Serial node !!!")

        self.send_datas = GimbalControl()
        self.lock = threading.Lock()
        self.serial_lock = threading.Lock()
        self.is_reconnecting = False
        
        # 创建qos
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        self.get_params()
        
        self.pub_uart_receive_decision = self.create_publisher(Decision, "/nav/decision", 10)
        self.sub_gimbal_control = self.create_subscription(GimbalControl, "tracker/gimbal_control", self.gimbal_control_callback, qos_profile_sensor_data)
        self.pub_uart_receive_imu = self.create_publisher(Vector3Stamped, '/imu/rpy', qos)

        self.color = ColorPrint()

        self.serial_receive_header = 0xA5
        self.serial_send_header = 0x5A

        # 初始化串口
        try:
            self.serial = serial.Serial(
                port = self.port_name,
                baudrate = self.baudrate,
                timeout = self.timeout,
                write_timeout = self.write_timeout,
            )
            if self.serial.is_open:
                self.get_logger().info(f"串口已打开: {self.port_name}")
                
                # 启动接收线程
                self.receive_thread = threading.Thread(target=self.receive_data, daemon=True)  
                self.receive_thread.start()
                
                # 启动独立的发送线程 (替代原来的 create_timer)
                self.send_thread = threading.Thread(target=self.send_data_loop, daemon=True)
                self.send_thread.start()
                
        except serial.SerialException as e:
            self.get_logger().error(f"创建串口时出错: {self.port_name} - {str(e)}")
            raise e    
            
    def get_params(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                ('port_name', '/dev/ttyACM0'),
                ('baudrate', 115200),
                ('timeout', 1.0),            
                ('write_timeout', 1.0),      
                ('flow_control', 'none'),
                ('parity', 'none'),
                ('stop_bits', '1'),
                ('serial_receive_header', 0xA5),
                ('serial_send_header', 0x5A)
            ]
        )

        self.port_name = self.get_parameter("port_name").value
        self.baudrate = self.get_parameter("baudrate").value
        self.timeout = self.get_parameter("timeout").value
        self.write_timeout = self.get_parameter("write_timeout").value
        self.flow_control = self.get_parameter("flow_control").value
        self.parity = self.get_parameter("parity").value
        self.stop_bits = self.get_parameter("stop_bits").value
        self.serial_receive_header = self.get_parameter("serial_receive_header").value
        self.serial_send_header = self.get_parameter("serial_send_header").value

        self.get_logger().info("-" * 30)
        self.get_logger().info("串口参数配置已加载:")
        self.get_logger().info(f"  端口号: {self.port_name}")
        self.get_logger().info(f"  波特率: {self.baudrate}")
        self.get_logger().info(f"  超时设置: Read={self.timeout}s, Write={self.write_timeout}s")
        self.get_logger().info(f"  校验/流控: Parity={self.parity}, Flow={self.flow_control}")
        self.get_logger().info("-" * 30)
 
    def gimbal_control_callback(self,msg):
        with self.lock:
            self.send_datas.pitch = msg.pitch
            self.send_datas.yaw = msg.yaw
            self.send_datas.can_fire = msg.can_fire

    def receive_data(self):
        packet_length = 16
        self.get_logger().info("接收数据线程已启动 (CRC16 Mode - 16 Bytes)")
        
        try:
            self.serial.reset_input_buffer()
        except Exception:
            pass

        while rclpy.ok():
            try:
                header = self.serial.read(1)
                if not header or header[0] != self.serial_receive_header:
                    continue

                remaining_data = self.serial.read(packet_length - 1)
                if len(remaining_data) != packet_length - 1:
                    self.get_logger().warn("数据包不完整")
                    continue

                full_packet = header + remaining_data
                data_payload = full_packet[:-2]
                checksum_bytes = full_packet[-2:]

                received_crc = struct.unpack('<H', checksum_bytes)[0]
                calculated_crc = get_crc16_check_sum(data_payload)
                
                if calculated_crc != received_crc:
                    continue

                _, detect_color, roll, pitch, yaw = struct.unpack("<BBfff", data_payload)

                # 发布 IMU 消息
                rpy_msg = Vector3Stamped()
                rpy_msg.header.stamp = self.get_clock().now().to_msg()
                rpy_msg.header.frame_id = 'imu_link'
                rpy_msg.vector.x = float(roll)
                rpy_msg.vector.y = float(pitch)
                rpy_msg.vector.z = float(yaw)
                self.pub_uart_receive_imu.publish(rpy_msg)

                # 发布 Decision 消息
                serial_decision_msg = Decision()
                serial_decision_msg.header.frame_id = 'serial_receive_frame'
                serial_decision_msg.header.stamp = self.get_clock().now().to_msg()
                serial_decision_msg.color = detect_color
                self.pub_uart_receive_decision.publish(serial_decision_msg)

            except (serial.SerialException, struct.error, ValueError, OSError) as e:
                self.get_logger().error(f"接收数据异常: {str(e)}")
                self.reopen_port()

    def send_data_loop(self):
        """专门用于发送数据的独立线程循环"""
        self.get_logger().info("发送数据线程已启动 (100Hz)")
        while rclpy.ok():
            self.Send()
            time.sleep(0.005) # 控制约 100Hz 的发送频率

    def Send(self):
        # 增加 serial_lock，防止重连时尝试向已关闭的对象写入数据
        with self.serial_lock:
            # 检查重连状态与串口对象有效性
            if self.is_reconnecting or not (hasattr(self, 'serial') and self.serial.is_open):
                return

            try:
                with self.lock:
                    # 显式转换数据类型，保障 struct.pack 不抛错
                    yaw = float(self.send_datas.yaw)
                    pitch = float(self.send_datas.pitch)
                    can_fire = int(self.send_datas.can_fire)
                    
                header = self.serial_send_header

                # 帧头(1), 云台yaw(4), 云台pitch(4), 开火(4)
                data_payload = struct.pack(
                    '<Bffi',
                    header,
                    yaw, 
                    pitch, 
                    can_fire, 
                )

                checksum = get_crc16_check_sum(data_payload)
                packet = data_payload + struct.pack("<H", checksum)
                self.serial.write(packet)
                # 添加这行打印调试信息，观察实际发出的 16 进制流
                # self.get_logger().info(f"发送报文: {' '.join([f'{b:02X}' for b in packet])}")

            except Exception as e:
                self.get_logger().error(f"发送数据时出错: {str(e)}")

    def reopen_port(self):
        with self.serial_lock:
            if self.is_reconnecting:
                return 
            self.is_reconnecting = True

        self.get_logger().warn("正在重连串口...")
        
        while rclpy.ok():
            try:
                # 在重新实例化之前获取锁，确保发送线程不会在实例化过程中引发崩溃
                with self.serial_lock:
                    if hasattr(self, 'serial') and self.serial and self.serial.is_open:
                        self.serial.close()
                    
                    self.serial = serial.Serial(
                        port=self.port_name,
                        baudrate=self.baudrate,
                        timeout=1,
                        write_timeout=1,
                    )
                self.get_logger().info("串口重连成功")
                break 
                
            except serial.SerialException as e:
                self.get_logger().error(f"串口重连失败，1秒后重试: {str(e)}")
                time.sleep(1)

        with self.serial_lock:
            self.is_reconnecting = False

def main(args=None):
    rclpy.init(args=args)
    node = SerialNode("serial_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Node terminated due to error: {e}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()