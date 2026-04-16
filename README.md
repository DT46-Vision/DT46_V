这是一版完整的 `README.md`，加入了你提供的 ROS 2 Humble 环境说明以及带有生成编译数据库指令的 `colcon` 编译命令。我还根据你代码中引用的库，补全了 Python 和 C++ 的常规依赖项，让你或者其他开发者克隆下来后能够直接“开箱即用”。

***

# DT-46-KielasVison
梓喵-铭

<br><img src="DT46-vision.svg" alt="DT46_vision" width="200" height="200">

<br>我的爱如潮水，you 然，我已无法再与它 apart

## 🔗 相关资料
* **[技术文档-GitHub](https://github.com/DT46-Vision/RM_Kielas_Vision_doc.git)**
* **[CNN训练代码-GitHub](https://github.com/DT46-Vision/KIELAS_Armor_CNN_Train.git)**

---

## 📖 项目简介
本项目为 RoboMaster 视觉自瞄系统，基于 ROS 2 框架开发。系统实现了从底层硬件通信、图像采集，到基于深度学习的装甲板识别，再到三维空间高频追踪与动态弹道解算的完整闭环。系统充分利用了 C++ 的高效性与 Python 的快速迭代优势。

### 🧠 核心算法亮点
* **Numba JIT 加速的 EKF (扩展卡尔曼滤波)**：构建了 9 维状态向量 `[xc, v_xc, yc, v_yc, za, v_za, yaw, v_yaw, r]`，引入马氏距离（Mahalanobis Distance）剔除 PnP 野值，保障高速运动下的状态收敛。
* **空气动力学弹道解算**：集成基于平方空气阻力模型 (F = -kv^2) 的欧拉数值积分打靶法，配合动态云台/枪口外参补偿，实现高精度动态击打。
* **反小陀螺与状态机机制**：动态监测目标角速度与装甲板跳变（Armor Jump），利用多块装甲板的拓扑关系进行平滑切换，有效应对敌方车辆的高频原地旋转。
* **无锁高性能渲染**：采用轻量级数据快照机制分离跟踪运算与 OpenCV 图像渲染，确保 EKF 高频迭代不受推流绘图的阻塞影响。

---

## ⚙️ 系统架构与模块划分

1. **rm_detector (C++)**
   * 装甲板检测与三维解算。传统视觉灯条提取结合 CNN 数字分类网络，输出目标 PnP 三维坐标。
2. **rm_tracker (Python)**
   * 目标追踪、位姿预测与发弹决策。接收相机与 IMU 位姿，进行坐标转换。内部维护 EKF 和状态机，解算提前量并下发云台控制指令。
3. **rm_serial & dm_imu (Python)**
   * 硬件通信层。解析达妙 IMU 的高频 RPY 数据，通过 CRC16 校验与下位机进行双向通讯。接收裁判系统状态（如颜色、弹速），下发云台解算结果。
4. **相机驱动层**
   * 支持海康工业相机 (`hik_camera`)、迈德威视相机 (`mindvision_camera`) 及普通 USB 摄像头 (`usb_camera`)。

---

## 📡 接口定义 (rm_interfaces)
系统内部的数据流转依赖自定义 ROS 2 消息类型：
* `ArmorsMsg`: 包含多块装甲板当前帧的三维坐标 (`dx, dy, dz`) 与相对偏航角 (`yaw`)。
* `ArmorsDebugMsg`: 包含装甲板的四点灯条像素坐标、颜色及 ID 等调试信息。
* `GimbalControl`: 包含解算后的云台目标姿态 (`pitch, yaw`) 与开火许可标志 (`can_fire`)。
* `Decision`: 包含裁判系统同步的比赛状态、目标颜色、实时弹速及血量信息。

---

## 🎯 装甲板类型定义

| 编号 | 含义             | 序号 |
|------|------------------|------|
| B1   | 蓝方1号 装甲板   | 0    |
| B2   | 蓝方2号 装甲板   | 1    |
| B3   | 蓝方3号 装甲板   | 2    |
| B4   | 蓝方4号 装甲板   | 3    |
| B5   | 蓝方5号 装甲板   | 4    |
| B7   | 蓝方哨兵 装甲板   | 5    |
| R1   | 红方1号 装甲板   | 6    |
| R2   | 红方2号 装甲板   | 7    |
| R3   | 红方3号 装甲板   | 8    |
| R4   | 红方4号 装甲板   | 9    |
| R5   | 红方5号 装甲板   | 10   |
| R7   | 红方哨兵 装甲板   | 11   |

---

## 🛠️ 部署与环境配置

### 1. 环境依赖
* **系统环境**: Ubuntu 22.04
* **ROS 2 版本**: Humble
* **Python 依赖库**:
  ```bash
  pip install numpy numba pyserial opencv-python
  ```
* **C++ 依赖库**: OpenCV, ONNXRuntime

### 2. 编译与运行
克隆工作空间后，在根目录执行以下命令进行编译：
```bash
# 生成 compile_commands.json 方便代码补全，并使用符号链接安装以加快 Python 节点调试
colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON --symlink-install

# 引入环境变量
source install/setup.bash
```

### 3. 串口定向映射 (Udev 规则)
本系统使用多串口并发通信，需配置 udev 规则绑定软链接，防止设备号飘移。

**获取串口信息**：
```bash
udevadm info -a -n /dev/ttyACM0 | grep -E '{idVendor}|{idProduct}|{serial}'
```
*(记录下输出结果中第一组的 `idVendor`、`idProduct` 和 `serial`)*

**编写 udev 规则**：
```bash
sudo nano /etc/udev/rules.d/99-fixed-serial.rules
```
填入以下规则（请根据实际硬件替换 serial）：
```bash
ACTION=="add", KERNEL=="ttyACM*", ATTRS{idVendor}=="ffff", ATTRS{idProduct}=="ffff", ATTRS{serial}=="2025021200", SYMLINK+="ttyPortIMU", MODE="0666"
ACTION=="add", KERNEL=="ttyACM*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5909029513", SYMLINK+="ttyPortMCU", MODE="0666"
```

**重新加载规则**：
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```
> ⚠️ **关键步骤**：配置完成后，请**务必手动将这两个串口的 USB 线从电脑上拔下并重新插入**。

---

### 4. 快捷启动配置 (Aliases)
建议在 `~/.bashrc` 中添加别名，实现一键拉起节点：
```bash
# nano ~/.bashrc
alias rmhiknav='ros2 launch rm_vision_bringup hik_nav.launch.py'
alias rmhik='ros2 launch rm_vision_bringup hik.launch.py'
alias rmmv='ros2 launch rm_vision_bringup mv.launch.py'
```

---

### 5. 场景启动说明 (Launch Files)
`rm_vision_bringup` 提供了多种启动脚本以适配不同兵种与测试场景：
* **实车部署 (全节点启动)**:
  * `ros2 launch rm_vision_bringup hik.launch.py`: 哨兵默认启动项 (海康相机 + 串口 + IMU + 检测 + 跟踪)。
  * `ros2 launch rm_vision_bringup mv.launch.py`: 步兵默认启动项 (迈德威视相机 + 串口 + IMU + 检测 + 跟踪)。
* **离线/视觉独立测试 (无串口通信)**:
  * `hik_nav.launch.py` / `mv_nav.launch.py` / `usb_nav.launch.py`: 仅启动相机、IMU及视觉算法层，用于纯视觉定位或录制数据集。
* **数据集回放**:
  * `ros2 launch rm_vision_bringup bag_record.launch.py bag_file:=/path/to/bag`: 读取 rosbag 数据并跑通视觉流，便于算法调参。