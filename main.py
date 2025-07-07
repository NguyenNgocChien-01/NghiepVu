
import argparse
import torch
import json
import os
from src.config import CONFIG
from src.model import InvoiceGCN
from src.data_processing import load_data
from src.train import run_training_session
from src.predict import load_inference_model, run_inference 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Công cụ trích xuất thông tin hóa đơn.")
    
    parser.add_argument(
        "--mode", "-m", 
        required=True, 
        choices=['train', 'predict'], 
        help="Chế độ hoạt động: 'train' để huấn luyện, 'predict' để dự đoán."
    )
    parser.add_argument(
        "--input", "-i", 
        type=int, 
        help="Chỉ số (index) của ảnh trong tập test cần dự đoán. Bắt buộc khi mode='predict'."
    )
    # THÊM MỚI: Tham số cho epochs
    parser.add_argument(
        "--epochs", "-e", 
        type=int, 
        default=2000, # Giá trị mặc định nếu không truyền vào
        help="Số lượng epochs để huấn luyện mô hình."
    )
    
    args = parser.parse_args()
    return args

def main():
    """Hàm chính để điều phối các tác vụ."""
    args = parse_arguments()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    if args.mode == 'train':
        print(f"--- Chế độ Huấn luyện với {args.epochs} epochs ---")
        
        # 1. Tải dữ liệu huấn luyện
        train_data, test_data = load_data(CONFIG)
        if train_data is None or test_data is None:
            return

        # 2. Cập nhật số epochs từ dòng lệnh vào config
        # Đây là bước quan trọng để truyền giá trị vào hàm huấn luyện
        train_config = CONFIG.copy()
        train_config['model_params']['num_epochs'] = args.epochs
        
        # 3. Gọi hàm huấn luyện
        run_training_session(train_config, train_data, test_data, device)
        
    elif args.mode == 'predict':
        if args.input is None:
            print("Lỗi: Cần cung cấp chỉ số ảnh với tham số --input (ví dụ: --input 5)")
            return
            
        print(f"--- Chế độ Dự đoán cho ảnh có chỉ số: {args.input} ---")
        
        # Tải dữ liệu và mô hình để dự đoán
        _, test_data = load_data(CONFIG)
        if test_data is None: return

        input_dim = test_data.num_node_features
        loaded_model = load_inference_model(input_dim, CONFIG, device)
        
        if loaded_model:
            run_inference(loaded_model, test_data, args.input, CONFIG, device)

if __name__ == "__main__":
    main()