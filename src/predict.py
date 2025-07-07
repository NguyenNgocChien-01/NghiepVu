import os
import shutil
import cv2
from src.config import CONFIG
from src.data_processing import Grapher
from src.model import InvoiceGCN
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm.notebook import tqdm as tqdm_nb
import json
def load_inference_model(input_dim, config, device):
    """
    Khởi tạo và tải state_dict cho mô hình từ file đã lưu.
    """
    print("Bắt đầu tải lại mô hình đã huấn luyện...")
    model_params = config['model_params']
    model = InvoiceGCN(
        input_dim=input_dim,
        hidden_dims=model_params['hidden_dims'],
        n_classes=model_params['n_classes'],
        dropout_rate=model_params['dropout_rate'],
        chebnet=model_params['chebnet'],
        K=model_params['K']
    )
    try:
        model.load_state_dict(torch.load(config['model_save_path'], map_location=device))
        model.to(device)
        model.eval()
        print(f"Đã tải mô hình từ '{config['model_save_path']}' thành công!")
        return model
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp mô hình tại '{config['model_save_path']}'.")
        return None

def get_image_details(img_id, config):
    """
    Lấy thông tin chi tiết (đồ thị, dataframe, ảnh) cho một ID ảnh.
    """
    connect = Grapher(filename=img_id, data_fd=config['data_folder'])
    G, _, df = connect.graph_formation() 
    img_path = os.path.join(config['data_folder'], "img", f"{img_id}.jpg")
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return G, df, img


# ==============================================================================
# PHẦN 2: HÀM SUY LUẬN CHÍNH (DÙNG INDEX)
# ==============================================================================


def run_inference(model, test_data, image_index, config, device):
    """
    Chạy suy luận, hiển thị và LƯU KẾT QUẢ cho một ảnh từ tập test.
    """
    # 1. Kiểm tra index và lấy thông tin
    num_images = len(test_data.img_id)
    if not (0 <= image_index < num_images):
        print(f"Lỗi: Chỉ số ảnh không hợp lệ. Vui lòng chọn một chỉ số từ 0 đến {num_images - 1}.")
        return

    single_graph_data = test_data.to_data_list()[image_index].to(device)
    img_id = single_graph_data.img_id
    
    print(f"\n--- Bắt đầu dự đoán cho ảnh: {img_id}.jpg (Chỉ số: {image_index}) ---")

    # 2. Chạy dự đoán
    with torch.no_grad():
        out = model(single_graph_data)
        pred_indices = out.max(dim=1)[1].cpu().numpy()

    # 3. Xử lý kết quả và vẽ lên ảnh
    label_map = {i: label for i, label in enumerate(config['labels'])}
    predicted_labels = [label_map.get(i, 'error') for i in pred_indices]
    
    grapher = Grapher(filename=img_id, data_fd=config['data_folder'])
    df_vis = grapher.get_df_for_visualization()
    image_to_draw = cv2.cvtColor(grapher.image, cv2.COLOR_BGR2RGB)

    df_vis['predicted_label'] = predicted_labels
    
    # Tạo dictionary để lưu thông tin dạng text
    extracted_info = defaultdict(list)

    for _, row in df_vis.iterrows():
        label = row['predicted_label']
        if label != 'other':
            # Thêm thông tin vào dictionary
            extracted_info[label.upper()].append(row['Object'])
            
            # Vẽ hộp và nhãn
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(image_to_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image_to_draw, label.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # In kết quả ra console
    print("\n--- KẾT QUẢ TRÍCH XUẤT ---")
    # Định dạng lại các trường nối liền nhau
    formatted_info = {}
    for key, value in extracted_info.items():
        if key in ['ADDRESS', 'COMPANY']:
            formatted_value = ' '.join(value)
        else:
            formatted_value = value
        print(f"{key}: {formatted_value}")
        formatted_info[key] = formatted_value


    

    output_dir = config["output_folder"]
    os.makedirs(output_dir, exist_ok=True)

    json_output_path = os.path.join(output_dir, f"{img_id}_result.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_info, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Đã lưu kết quả dạng text tại: {json_output_path}")

    #Lưu ảnh đã vẽ bounding box và hiển thị
    img_output_path = os.path.join(output_dir, f"{img_id}_annotated.png")
    
    plt.figure(figsize=(10, 15))
    plt.imshow(image_to_draw)
    plt.title(f"Kết quả dự đoán cho ảnh: {img_id}")
    plt.axis('off')
    
    # Lưu ảnh trước khi hiển thị
    plt.savefig(img_output_path, bbox_inches='tight')
    print(f"✅ Đã lưu ảnh kết quả tại: {img_output_path}")
    
    plt.show() # Vẫn hiển thị ảnh
    plt.close() # Đóng figure để giải phóng bộ nhớ