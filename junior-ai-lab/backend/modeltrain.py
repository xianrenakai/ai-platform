#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN模型训练模板 - 用于卷积神经网络模型训练
学生使用说明：
1. 从"模型设计"页面复制生成的模型代码，粘贴到下方 model_code 变量中
2. 设置训练轮次 epochs（建议5-20轮）
3. 设置数据集名称 dataset_name（从"创建数据集"页面获取）
4. 运行此脚本，开始训练模型
5. 训练完成后，将生成的.h5模型文件上传到"模型测试"页面
"""

# ============ 配置区域 - 请修改以下参数 ============

# 1. 模型代码（从"模型设计"页面复制）
model_code = """
# 在这里粘贴你的模型代码
# 例如：
# from tensorflow import keras
# from tensorflow.keras import layers
# 
# model = keras.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])
"""

# 2. 训练轮次（建议5-20）
epochs = 10

# 3. 数据集名称（从"创建数据集"页面获取）
dataset_name = "my_mnist_dataset"

# 4. 批次大小（一般使用32或64）
batch_size = 32

# 5. 模型保存名称（可自定义）
model_save_name = "my_cnn_model.h5"

# ============ 以下代码无需修改 ============

import os
import sys
import numpy as np
from pathlib import Path

try:
    from tensorflow import keras
    from tensorflow.keras import datasets
    import tensorflow as tf
except ImportError:
    print("错误：未安装 TensorFlow")
    print("请运行：pip install tensorflow")
    sys.exit(1)


def load_dataset(dataset_name):
    """加载数据集"""
    print(f"\n正在加载数据集: {dataset_name}")
    
    # 这里使用MNIST作为示例，实际应该加载学生创建的数据集
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    
    # 数据预处理
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    print(f"训练集大小: {x_train.shape[0]}")
    print(f"测试集大小: {x_test.shape[0]}")
    
    return (x_train, y_train), (x_test, y_test)


def create_model(model_code):
    """根据学生代码创建模型"""
    print("\n正在创建模型...")
    
    try:
        # 执行模型代码
        local_vars = {}
        exec(model_code, globals(), local_vars)
        
        if 'model' not in local_vars:
            raise ValueError("模型代码中未找到 'model' 变量")
        
        model = local_vars['model']
        return model
    
    except Exception as e:
        print(f"模型创建失败: {str(e)}")
        print("\n请检查你的模型代码是否正确")
        sys.exit(1)


def train_model(model, train_data, test_data, epochs, batch_size):
    """训练模型"""
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    
    print("\n编译模型...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n模型结构:")
    model.summary()
    
    print(f"\n开始训练 (共{epochs}轮)...")
    print("=" * 60)
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("训练完成!")
    
    return history


def evaluate_model(model, test_data):
    """评估模型"""
    x_test, y_test = test_data
    
    print("\n正在评估模型...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\n最终测试结果:")
    print(f"  测试准确率: {test_acc * 100:.2f}%")
    print(f"  测试损失: {test_loss:.4f}")
    
    return test_acc, test_loss


def save_model(model, filename):
    """保存模型"""
    print(f"\n正在保存模型到: {filename}")
    model.save(filename)
    print(f"模型已保存!")
    print(f"请在'模型测试'页面上传此文件: {os.path.abspath(filename)}")


def main():
    """主函数"""
    print("=" * 60)
    print("CNN模型训练程序")
    print("=" * 60)
    
    # 检查配置
    if "在这里粘贴你的模型代码" in model_code:
        print("\n错误：请先粘贴你的模型代码!")
        print("从'模型设计'页面复制代码，粘贴到 model_code 变量中")
        return
    
    # 加载数据集
    train_data, test_data = load_dataset(dataset_name)
    
    # 创建模型
    model = create_model(model_code)
    
    # 训练模型
    history = train_model(model, train_data, test_data, epochs, batch_size)
    
    # 评估模型
    test_acc, test_loss = evaluate_model(model, test_data)
    
    # 保存模型
    save_model(model, model_save_name)
    
    print("\n" + "=" * 60)
    print("训练流程完成!")
    print("=" * 60)
    print("\n下一步：")
    print("1. 在'模型测试'页面上传生成的 .h5 文件")
    print("2. 开始测试你的模型!")


if __name__ == "__main__":
    main()
