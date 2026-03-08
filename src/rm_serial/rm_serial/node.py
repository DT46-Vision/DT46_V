import time
import serial
import threading
import struct
import rclpy
from .modules.crc import *
from rclpy.node import Node
from std_msgs.msg import Header
from rm_interfaces.msg import GimbalControl, Decision
from geometry_msgs.msg import Vector3Stamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class ColorPrint():
    def __init__(self):
        self.PINK = "\033[38;5;218m"
        self.CYAN = "\033[96m"
        self.GREEN = "\033[32m"
        self.RED = "\033[31m"
        self.BLUE = "\033[34m"
        self.RESET = "\033[0m"

class RMSerialDriver(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("启动 RMSerialDriver (CRC32 Mode)!")

        # 获取参数
        self.get_params()

        # 创建订阅者
        self.sub_gimbal_control = self.create_subscription(
            GimbalControl, "/tracker/gimbal_control", self.send_data, 10
        )

        # 创建发布者 1: 决策信息
        self.pub_uart_receive_decision = self.create_publisher(Decision, "/nav/decision", 10)

        # ---------- QoS ----------
        # IMU 数据通常需要高实时性，Volatile 是正确的选择
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            durability=DurabilityPolicy.VOLATILE,
        )
        # 创建发布者 2: IMU 数据
        self.pub_uart_receive_imu = self.create_publisher(Vector3Stamped, '/imu/rpy', qos)

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
                self.get_logger().info(f"串口 {self.device_name} 打开成功")
                self.receive_thread = threading.Thread(target=self.receive_data)
                self.receive_thread.setDaemon(True) # 设置为守护线程
                self.receive_thread.start()
        except serial.SerialException as e:
            self.get_logger().error(f"创建串口时出错: {self.device_name} - {str(e)}")
            raise e

    def get_params(self):
        self.declare_parameter("device_name", "/dev/ttyUSB0")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("flow_control", "none")
        self.declare_parameter("parity", "none")
        self.declare_parameter("stop_bits", "1")

        self.device_name = self.get_parameter("device_name").value
        self.baud_rate = self.get_parameter("baud_rate").value
        self.flow_control = self.get_parameter("flow_control").value
        self.parity = self.get_parameter("parity").value
        self.stop_bits = self.get_parameter("stop_bits").value

    def receive_data(self):
        serial_receive_msg = Decision()
        serial_receive_msg.header.frame_id = 'serial_receive_frame'
        # 总长 16 = Header(1)+Color(1)+Roll(4)+Pitch(4)+Yaw(4) + CRC(2)
        packet_length = 16

        # 建立本地内存缓冲区
        buffer = bytearray()

        self.get_logger().info("接收数据线程已启动 (CRC16 Mode - 16 Bytes)")

        while rclpy.ok():
            try:
                # 1. 一次性读取底层积压的所有字节
                waiting = self.serial_port.in_waiting
                if waiting > 0:
                    buffer.extend(self.serial_port.read(waiting))
                else:
                    # 避免 CPU 空转
                    time.sleep(0.001)
                    continue

                # 2. 在内存中循环处理所有完整的包
                while len(buffer) >= packet_length:
                    # 寻找帧头 0xA5 (即十进制 165)
                    if buffer[0] == 0xA5:
                        # 提取一个完整包
                        full_packet = bytes(buffer[:packet_length])

                        data_payload = full_packet[:-2]
                        checksum_bytes = full_packet[-2:]
                        received_crc = struct.unpack('<H', checksum_bytes)[0]
                        calculated_crc = get_crc16_check_sum(data_payload)

                        if calculated_crc == received_crc:
                            # 校验通过，解包
                            _, detect_color, roll, pitch, yaw = struct.unpack("<BBfff", data_payload)

                            rpy_msg = Vector3Stamped()
                            rpy_msg.header.stamp = self.get_clock().now().to_msg()
                            rpy_msg.header.frame_id = 'imu_link'
                            rpy_msg.vector.x = float(roll)
                            rpy_msg.vector.y = float(pitch)
                            rpy_msg.vector.z = float(yaw)
                            self.pub_uart_receive_imu.publish(rpy_msg)

                            serial_receive_msg.header.stamp = self.get_clock().now().to_msg()
                            serial_receive_msg.color = detect_color
                            self.pub_uart_receive_decision.publish(serial_receive_msg)

                            # 从缓冲区移除已处理的合法包
                            del buffer[:packet_length]
                        else:
                            # 校验失败：说明这个 0xA5 可能是数据段里的伪造帧头，只丢弃一个字节，继续往后找
                            del buffer[:1]
                    else:
                        # 帧头不对，丢弃首字节，继续向后滑动
                        del buffer[:1]

            except (serial.SerialException, struct.error, ValueError, OSError) as e:
                self.get_logger().error(f"接收数据异常: {str(e)}")
                self.reopen_port()

    def send_data(self, msg):
        try:
            header = 0x5A
            pitch  = msg.pitch
            yaw    = -msg.yaw
            shoot  = msg.can_fire
            # 1. 打包数据载荷 (10字节)
            # <BffB: Header(1), Pitch(4), Yaw(4), Shoot(1)
            # 这里的顺序必须严格对应
            data_payload = struct.pack("<BffB", header, pitch, yaw, shoot)

            # 2. 计算 CRC16 校验码
            # 注意：是对前 10 个字节(data_payload) 计算 CRC
            checksum = get_crc16_check_sum(data_payload)

            # 3. 组合完整包 (12字节)
            # 将 checksum 打包成 2字节无符号短整型 (<H) 并拼接到末尾
            packet = data_payload + struct.pack("<H", checksum)

            self.serial_port.write(packet)
        except Exception as e:
            self.get_logger().error(f"发送数据异常: {str(e)}")

    def reopen_port(self):
        self.get_logger().warn("正在重连串口...")
        try:
            if self.serial_port.is_open:
                self.serial_port.close()
            self.serial_port.open()
            self.get_logger().info("串口重连成功")
        except Exception as e:
            self.get_logger().error("串口重连失败，1秒后重试")
            time.sleep(1)
            self.reopen_port()

def main(args=None):
    rclpy.init(args=args)
    node = RMSerialDriver("rm_serial")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
