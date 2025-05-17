import pandas as pd
import numpy as np
from PIL import Image
import os

def resize_image(folder, size=(160,120)):
    '''
    Chuẩn hóa kích thước của ảnh
    - Đầu vào: đường dẫn của ảnh
    - Đầu ra: lưu lại ảnh đã được chuẩn hóa

    Hiện tại trong dataset có 2 loại ảnh: size 160x120(bộ k có human noise) và size 320x240(bộ có human noise) 
    --> chuẩn hóa về size 160x120
    '''

    for file_name in os.listdir(folder):
        if file_name.endswith(('jpg','png', 'jpeg')):
            img_path = os.path.join(folder, file_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(size)
                img.save(img_path)
                print(f"Resized: {file_name}")
            except Exception as e:
                print(f"Error with {file_name}: {e}")
    
def load_image(img_path, mode='rgb'):
    '''
    Hàm load ảnh
    - Đầu vào: đường dẫn ảnh
    - Đầu ra: ảnh
    '''
    img = Image.open(img_path)
    if mode == 'gray':
        img = img.convert('L') #Dùng trong trường hợp ML, số chiều quá nhiều
    else:
        img = img.convert('RGB')
    return img

def normalize_pixels(img):
    '''
    Chuẩn hóa pixel
    - Đầu vào: ảnh
    - Đầu ra: ảnh đã được chuẩn hóa
    '''

    return np.array(img).astype('float32') / 255.0

def flatten_img(img):
    '''
    Chuyển vector ảnh từ 3D sang 1D
    '''
    return normalize_pixels(img).all