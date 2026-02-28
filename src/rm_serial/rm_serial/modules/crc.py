# crc.py
from .crc_table import CRC8_TAB, CRC16_TAB, CRC8_INIT, CRC16_INIT

# ==========================================
# CRC8 实现
# ==========================================

def get_crc8_check_sum(message: bytes, crc_init: int = CRC8_INIT) -> int:
    """
    计算 CRC8 校验值。
    对应 C 函数: Get_CRC8_Check_Sum
    :param message: 数据内容 (bytes 或 bytearray)
    :param crc_init: 初始值 (默认使用 CRC8_INIT)
    :return: 8位 CRC 校验码 (int)
    """
    crc = crc_init
    for byte in message:
        index = crc ^ byte
        crc = CRC8_TAB[index]
    return crc

def verify_crc8_check_sum(message: bytes) -> bool:
    """
    校验 CRC8。
    对应 C 函数: Verify_CRC8_Check_Sum
    假设数据的最后 1 个字节是校验位。
    """
    if not message or len(message) <= 2:
        return False

    # 取出数据部分（除去最后1字节）进行计算
    expected = get_crc8_check_sum(message[:-1], CRC8_INIT)

    # 比较计算值与实际值
    return expected == message[-1]

def append_crc8_check_sum(message: bytearray) -> None:
    """
    在数据末尾追加 CRC8 校验位。
    对应 C 函数: Append_CRC8_Check_Sum
    注意：传入的 bytearray 长度必须包含校验位的占位符。
    该函数会原地修改 message 的最后一个字节。
    """
    if not message or len(message) <= 2:
        return

    # 计算除最后一位外的 CRC
    crc = get_crc8_check_sum(message[:-1], CRC8_INIT)
    message[-1] = crc


# ==========================================
# CRC16 实现
# ==========================================

def get_crc16_check_sum(message: bytes, crc_init: int = CRC16_INIT) -> int:
    """
    计算 CRC16 校验值。
    对应 C 函数: Get_CRC16_Check_Sum
    :param message: 数据内容 (bytes 或 bytearray)
    :param crc_init: 初始值 (默认使用 CRC16_INIT)
    :return: 16位 CRC 校验码 (int)
    """
    crc = crc_init
    for byte in message:
        # C 逻辑: (wCRC >> 8) ^ wCRC_Table[(wCRC ^ chData) & 0xff]
        index = (crc ^ byte) & 0xff
        crc = (crc >> 8) ^ CRC16_TAB[index]
    return crc

def verify_crc16_check_sum(message: bytes) -> bool:
    """
    校验 CRC16。
    对应 C 函数: Verify_CRC16_Check_Sum
    假设数据的最后 2 个字节是校验位 (小端序)。
    """
    if not message or len(message) <= 2:
        return False

    # 计算除最后两字节外的 CRC
    expected = get_crc16_check_sum(message[:-2], CRC16_INIT)

    # 提取末尾的实际校验位 (Little Endian: 低位在前)
    actual_low = message[-2]
    actual_high = message[-1]

    # 比较
    return (expected & 0xff) == actual_low and ((expected >> 8) & 0xff) == actual_high

def append_crc16_check_sum(message: bytearray) -> None:
    """
    在数据末尾追加 CRC16 校验位 (小端序)。
    对应 C 函数: Append_CRC16_Check_Sum
    注意：传入的 bytearray 长度必须包含2个字节的占位符。
    该函数会原地修改 message 的最后两个字节。
    """
    if not message or len(message) <= 2:
        return

    # 计算除最后两位外的 CRC
    crc = get_crc16_check_sum(message[:-2], CRC16_INIT)

    # 填入低字节
    message[-2] = crc & 0xff
    # 填入高字节
    message[-1] = (crc >> 8) & 0xff
