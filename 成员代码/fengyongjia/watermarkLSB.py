# encoding=utf-8
"""
改进LSB图像隐写算法实现
功能：使用次低有效位（bit-1）+ 随机位置嵌入隐藏信息，支持信息提取与PSNR质量评估
安全说明：本代码仅用于学习信息隐写技术，请勿用于非法数据隐藏
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math

def text_to_binary(text):
    """将文本字符串转换为8位二进制字符串"""
    binary_str = ''
    for char in text:
        # 每个字符转换为8位二进制，不足补0
        binary_str += format(ord(char), '08b')
    return binary_str

def binary_to_text(binary_str):
    """将8位对齐的二进制字符串转换回文本"""
    text = ''
    # 按每8位分割为一个字节
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        text += chr(int(byte, 2))
    return text

def calculate_psnr_grayscale(img1_path, img2_path):
    """
    计算两幅灰度图像的峰值信噪比（PSNR）
    :param img1_path: 原始图像路径
    :param img2_path: 携密图像路径
    :return: PSNR值（dB）
    """
    # 以灰度模式读取图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    # 图像读取异常处理
    if img1 is None or img2 is None:
        raise ValueError("错误：无法读取图像文件，请检查文件路径是否正确")
    
    # 统一图像尺寸
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # 转换为浮点型避免计算溢出
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    
    # 计算均方误差
    mse = np.mean((img1 - img2) ** 2)
    
    # 图像完全一致时直接返回满分PSNR
    if mse < 1.0e-10:
        return 100
    
    PIXEL_MAX = 255.0
    psnr_value = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return round(psnr_value, 2)

class ImprovedLSB:
    """改进型LSB隐写实现类：次低有效位嵌入 + 固定随机种子位置选择"""
    def __init__(self):
        # 固定随机种子，保证嵌入与提取位置一致
        self.RANDOM_SEED = 2021

    def hide_message_improved_sequential(self, original_path, secret_msg, output_path):
        """
        改进LSB隐写：次低有效位(bit1) + 随机位置嵌入
        :param original_path: 原始载体图像路径
        :param secret_msg: 待隐藏文本信息
        :param output_path: 输出携密图像路径
        """
        # 读取并统一为RGB图像
        img = Image.open(original_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_array = np.array(img)
        height, width, channels = img_array.shape
        
        # 转换消息为二进制
        binary_msg = text_to_binary(secret_msg)
        msg_length = len(binary_msg)
        
        # 容量校验
        if msg_length > height * width:
            raise ValueError("错误：隐藏信息超出图像可承载容量")
        
        # 复制图像数组，仅修改R通道
        stego_array = img_array.copy()
        flat_r_channel = stego_array[:, :, 0].flatten()
        
        # 生成固定随机嵌入位置
        np.random.seed(self.RANDOM_SEED)
        positions = np.random.permutation(height * width)[:msg_length]
        
        # 次低有效位（bit1）嵌入
        for i, idx in enumerate(positions):
            # 清除bit1，再写入秘密位
            flat_r_channel[idx] = (flat_r_channel[idx] & 0xFD) | (int(binary_msg[i]) << 1)
        
        # 重构图像并保存
        stego_array[:, :, 0] = flat_r_channel.reshape((height, width))
        stego_img = Image.fromarray(stego_array)
        stego_img.save(output_path)
        
        return img_array, stego_array, msg_length

    def extract_message_improved_sequential(self, stego_path, msg_length_bits):
        """从携密图像中提取隐藏信息"""
        stego_img = Image.open(stego_path)
        stego_array = np.array(stego_img)
        flat_r_channel = stego_array[:, :, 0].flatten()
        binary_msg = ''
        
        # 重建随机位置
        height, width = stego_array.shape[:2]
        np.random.seed(self.RANDOM_SEED)
        positions = np.random.permutation(height * width)[:msg_length_bits]
        
        # 从bit1提取数据
        for idx in positions:
            bit_value = (flat_r_channel[idx] >> 1) & 1
            binary_msg += str(bit_value)
        
        secret_msg = binary_to_text(binary_msg)
        return binary_msg, secret_msg

def display_improved_comparison(original_array, stego_array):
    """显示原始图像与携密图像对比"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(original_array)
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(stego_array)
    plt.title('改进LSB携密图像')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 主程序入口
if __name__ == "__main__":
    print("\n=== 改进的LSB隐写实验（随机位嵌入） ===")
    try:
        improved_lsb = ImprovedLSB()
        
        # 信息隐藏
        original_improved, stego_improved, msg_length_improved = (
            improved_lsb.hide_message_improved_sequential(
                'buptgray.bmp', "BUPTshahexiaoqu", 'buptgraystego1.bmp'
            )
        )
        
        # 显示对比效果
        display_improved_comparison(original_improved, stego_improved)
        
        # 信息提取
        binary_improved, extracted_improved = (
            improved_lsb.extract_message_improved_sequential(
                'buptgraystego1.bmp', msg_length_improved
            )
        )
        
        print("提取隐藏的秘密信息二进制为: " + binary_improved)
        print("提取隐藏的秘密信息为: " + extracted_improved)
        
        # 图像质量评估
        psnr_improved = calculate_psnr_grayscale('buptgray.bmp', 'buptgraystego1.bmp')
        print("本组计算的峰值信噪比为: " + str(psnr_improved))
        
        print("\n实验执行完成！")

    except Exception as e:
        print(f"执行出错: {e}")
