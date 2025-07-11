import os
import shutil
import cv2
from src.config import CONFIG
from src.data_processing import Grapher, load_data
from src.model import InvoiceGCN
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm.notebook import tqdm as tqdm_nb
import json
from datetime import datetime
import csv
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import time
import re

def write_evaluation_log(img_id, config, extracted_info, img_path, y_true, y_pred, type_run="predict", seed_value=42, search_already=False, status="Success", error_message=None, val_loss=0.0, processing_time=0.0):
    """
    Ghi thông tin đánh giá hiệu quả của mô hình vào file log .csv để phân tích, bao gồm chi tiết cho từng nhãn.
    
    Args:
        img_id (str): ID của ảnh hóa đơn.
        config (dict): Cấu hình từ config.py.
        extracted_info (dict): Thông tin trích xuất dạng dictionary.
        img_path (str): Đường dẫn file ảnh.
        y_true (np.array): Nhãn thực tế.
        y_pred (np.array): Nhãn dự đoán.
        type_run (str): Loại chạy (predict hoặc train).
        seed_value (int): Giá trị seed khởi tạo.
        search_already (bool): Trạng thái kiểm tra xem ảnh đã được xử lý trước đó chưa.
        status (str): Trạng thái thực hiện (Success hoặc Error).
        error_message (str): Thông báo lỗi nếu có.
        val_loss (float): Loss trên tập test.
        processing_time (float): Thời gian xử lý đánh giá.
    """
    log_dir = os.path.join(config['output_folder'], "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Xác định tên file dựa trên company và recall
    company = extracted_info.get('COMPANY', 'unknown').replace(" ", "_").replace("&", "and").replace(".", "").replace(",", "")[:50]
    report = {}
    recall = "0.0000"
    if status == "Success" and len(y_true) > 0 and len(y_pred) > 0:
        target_names = config['labels']
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True)
        recall = f"{report['macro avg']['recall']:.4f}"
    log_path = os.path.join(log_dir, f"{company}_{recall}.csv")
    
    # Khởi tạo các giá trị mặc định
    val_acc, val_auc, val_mmc = 0.0, 0.0, 0.0
    val_acc_a, sd_acc = 0.0, 0.0
    val_auc_a, va_mmc_a, sd_mmc = 0.0, 0.0, 0.0
    ep_stopped = config['model_params']['num_epochs']
    index_fold = 0
    
    # Nếu không có lỗi, tính toán các độ đo hiệu quả
    if status == "Success" and len(y_true) > 0 and len(y_pred) > 0:
        target_names = config['labels']
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True)
        
        # Tính ma trận nhầm lẫn
        cm = confusion_matrix(y_true, y_pred)
        total_samples = cm.sum()
        val_acc = report['accuracy']
        
        # Tính AUC
        try:
            y_true_one_hot = np.eye(len(target_names))[y_true]
            y_score = np.eye(len(target_names))[y_pred]
            val_auc = roc_auc_score(y_true_one_hot, y_score, multi_class='ovr', average='macro')
        except:
            val_auc = 0.0
        
        # Tính MMC tổng quát
        val_mmc = np.mean(np.diag(cm)) / total_samples if total_samples > 0 else 0.0
        
        # Giả định trung bình và độ lệch chuẩn
        val_acc_a, sd_acc = val_acc, 0.0
        val_auc_a = val_auc
        va_mmc_a, sd_mmc = val_mmc, 0.0
    
    # Chuẩn bị dữ liệu cho CSV
    rows = []
    # Performance Metrics cho Macro Avg lên đầu
    if status == "Success" and report and len(y_true) > 0 and len(y_pred) > 0:
        rows.append({
            "Label": "Macro Avg",
            "Precision": f"{report['macro avg']['precision']:.4f}",
            "Recall": f"{report['macro avg']['recall']:.4f}",
            "F1-Score": f"{report['macro avg']['f1-score']:.4f}",
            "Val Acc": val_acc,
            "Val AUC": val_auc,
            "Val Loss": val_loss,
            "Val MMC": val_mmc,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "TP": 0,
            "Time": processing_time,
            "Ep Stopped": ep_stopped,
            "Index Fold": index_fold,
            "Val Acc Avg": val_acc_a,
            "SD Acc": sd_acc,
            "Val AUC Avg": val_auc_a,
            "SD MMC": sd_mmc,
            "Va MMC Avg": va_mmc_a
        })
        
        # Performance Metrics cho từng nhãn
        cm = confusion_matrix(y_true, y_pred)
        for i, label in enumerate(target_names):
            # Tính TN, FP, FN, TP cho từng nhãn (one-vs-rest)
            true_pos = cm[i, i] if i < cm.shape[0] else 0
            false_pos = cm[:, i].sum() - true_pos
            false_neg = cm[i, :].sum() - true_pos
            true_neg = total_samples - (true_pos + false_pos + false_neg)
            
            # Tính Val MMC cho nhãn
            val_mmc_label = true_pos / (true_pos + false_pos + false_neg) if (true_pos + false_pos + false_neg) > 0 else 0.0
            
            # Tính Val AUC cho nhãn (one-vs-rest)
            try:
                y_true_binary = (y_true == i).astype(int)
                y_pred_binary = (y_pred == i).astype(int)
                val_auc_label = roc_auc_score(y_true_binary, y_pred_binary)
            except:
                val_auc_label = 0.0
            
            rows.append({
                "Label": label,
                "Precision": f"{report[label]['precision']:.4f}",
                "Recall": f"{report[label]['recall']:.4f}",
                "F1-Score": f"{report[label]['f1-score']:.4f}",
                "Val Acc": val_acc,
                "Val AUC": val_auc_label,
                "Val Loss": val_loss,
                "Val MMC": val_mmc_label,
                "TN": true_neg,
                "FP": false_pos,
                "FN": false_neg,
                "TP": true_pos,
                "Time": "",
                "Ep Stopped": "",
                "Index Fold": "",
                "Val Acc Avg": "",
                "SD Acc": "",
                "Val AUC Avg": "",
                "SD MMC": "",
                "Va MMC Avg": ""
            })
    else:
        rows.append({
            "Label": status if status != "Success" else "",
            "Precision": error_message if error_message else "",
            "Recall": "",
            "F1-Score": "",
            "Val Acc": val_acc,
            "Val AUC": val_auc,
            "Val Loss": val_loss,
            "Val MMC": val_mmc,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "TP": 0,
            "Time": processing_time,
            "Ep Stopped": ep_stopped,
            "Index Fold": index_fold,
            "Val Acc Avg": val_acc_a,
            "SD Acc": sd_acc,
            "Val AUC Avg": val_auc_a,
            "SD MMC": sd_mmc,
            "Va MMC Avg": va_mmc_a
        })
    
    # Extracted Information
    for key, value in extracted_info.items():
        rows.append({
            "Label": key,
            "Precision": str(value),
            "Recall": "",
            "F1-Score": "",
            "Val Acc": "",
            "Val AUC": "",
            "Val Loss": "",
            "Val MMC": "",
            "TN": "",
            "FP": "",
            "FN": "",
            "TP": "",
            "Time": "",
            "Ep Stopped": "",
            "Index Fold": "",
            "Val Acc Avg": "",
            "SD Acc": "",
            "Val AUC Avg": "",
            "SD MMC": "",
            "Va MMC Avg": ""
        })
    
    # Ghi vào file CSV
    fieldnames = ["Label", "Precision", "Recall", "F1-Score", "Val Acc", "Val AUC", "Val Loss", "Val MMC", "TN", "FP", "FN", "TP", "Time", "Ep Stopped", "Index Fold", "Val Acc Avg", "SD Acc", "Val AUC Avg", "SD MMC", "Va MMC Avg"]
    with open(log_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"✅ Đã lưu kết quả đánh giá tại: {log_path}")

def load_inference_model(input_dim, config, device):
    """
    Khởi tạo và tải state_dict cho mô hình từ file đã lưu để đánh giá.
    """
    print("Bắt đầu tải lại mô hình đã huấn luyện để đánh giá...")
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
    Lấy thông tin chi tiết (đồ thị, dataframe, ảnh) cho một ID ảnh để đánh giá.
    """
    connect = Grapher(filename=img_id, data_fd=config['data_folder'])
    G, _, df = connect.graph_formation() 
    img_path = os.path.join(config['data_folder'], "img", f"{img_id}.jpg")
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return G, df, img, img_path

def run_inference(model, test_data, image_index, config, device):
    """
    Chạy đánh giá hiệu quả của mô hình và lưu kết quả cho một ảnh từ tập test.
    """
    # Đo thời gian xử lý
    start_time = time.time()
    
    # 1. Kiểm tra index và lấy thông tin
    num_images = len(test_data.img_id)
    if not (0 <= image_index < num_images):
        print(f"Lỗi: Chỉ số ảnh không hợp lệ. Vui lòng chọn một chỉ số từ 0 đến {num_images - 1}.")
        write_evaluation_log("unknown", config, {}, "", [], [], type_run="predict", seed_value=42, search_already=False, 
                  status="Error", error_message="Invalid image index", val_loss=0.0, processing_time=0.0)
        return

    single_graph_data = test_data.to_data_list()[image_index].to(device)
    img_id = single_graph_data.img_id
    
    print(f"\n--- Bắt đầu đánh giá hiệu quả cho ảnh: {img_id}.jpg (Chỉ số: {image_index}) ---")

    # 2. Chạy dự đoán để đánh giá
    try:
        with torch.no_grad():
            out = model(single_graph_data)
            pred_indices = out.max(dim=1)[1].cpu().numpy()

        # 3. Xử lý kết quả và vẽ lên ảnh để hỗ trợ đánh giá
        label_map = {i: label for i, label in enumerate(config['labels'])}
        predicted_labels = [label_map.get(i, 'error') for i in pred_indices]
        
        grapher = Grapher(filename=img_id, data_fd=config['data_folder'])
        df_vis = grapher.get_df_for_visualization()
        image_to_draw = cv2.cvtColor(grapher.image, cv2.COLOR_BGR2RGB)

        df_vis['predicted_label'] = predicted_labels
        
        # Tạo dictionary để lưu thông tin trích xuất hỗ trợ đánh giá
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
        print("\n--- KẾT QUẢ TRÍCH XUẤT HỖ TRỢ ĐÁNH GIÁ ---")
        # Định dạng lại các trường nối liền nhau
        formatted_info = {}
        for key, value in extracted_info.items():
            if key in ['ADDRESS', 'COMPANY']:
                formatted_value = ' '.join(value)
            else:
                formatted_value = value
            print(f"{key}: {formatted_value}")
            formatted_info[key] = formatted_value

        # Lấy đường dẫn ảnh và nhãn thực tế
        _, _, _, img_path = get_image_details(img_id, config)
        y_true = single_graph_data.y.cpu().numpy() - 1  # Điều chỉnh nhãn bắt đầu từ 0
        y_pred = pred_indices

        # Tính val_loss
        val_loss = F.nll_loss(out, torch.tensor(y_true, device=device)).item()

        # Kiểm tra search_already
        json_output_path = os.path.join(config["output_folder"], f"{img_id}_result.json")
        search_already = os.path.exists(json_output_path)

        # Tính thời gian xử lý
        processing_time = time.time() - start_time

        # Ghi log đánh giá
        write_evaluation_log(img_id, config, formatted_info, img_path, y_true, y_pred, 
                  type_run="predict", seed_value=42, search_already=search_already, 
                  val_loss=val_loss, processing_time=processing_time)

        output_dir = config["output_folder"]
        os.makedirs(output_dir, exist_ok=True)

        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_info, f, ensure_ascii=False, indent=4)
        print(f"\n✅ Đã lưu kết quả dạng text tại: {json_output_path}")

        # Lưu ảnh đã vẽ bounding box và hiển thị
        img_output_path = os.path.join(output_dir, f"{img_id}_annotated.png")
        
        plt.figure(figsize=(10, 15))
        plt.imshow(image_to_draw)
        plt.title(f"Kết quả đánh giá cho ảnh: {img_id}")
        plt.axis('off')
        
        # Lưu ảnh trước khi hiển thị
        plt.savefig(img_output_path, bbox_inches='tight')
        print(f"✅ Đã lưu ảnh kết quả tại: {img_output_path}")
        
        plt.show()  # Vẫn hiển thị ảnh
        plt.close()  # Đóng figure để giải phóng bộ nhớ

    except Exception as e:
        print(f"Lỗi trong quá trình đánh giá: {str(e)}")
        _, _, _, img_path = get_image_details(img_id, config) if 'img_id' in locals() else ("", "", "", "")
        write_evaluation_log(img_id if 'img_id' in locals() else "unknown", config, {}, img_path, [], [], 
                  type_run="predict", seed_value=42, search_already=False, 
                  status="Error", error_message=str(e), val_loss=0.0, processing_time=time.time() - start_time)