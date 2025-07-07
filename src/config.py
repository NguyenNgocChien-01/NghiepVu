# src/config.py
CONFIG = {
    # Đường dẫn
    "data_folder": "data/raw",
    "processed_folder": "data/processed",
    "model_save_path": "outputs/models/kie_gcn_model_best.pth",
    "output_folder": "outputs/results",

    # Cấu hình mô hình
    "model_params": {
        'lr': 0.001,
        'weight_decay': 0.001,
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.3,
        'n_classes': 6,
        'num_epochs': 2000, # Đây là số epoch đã huấn luyện
        'chebnet': True,
        'K': 3
    },
    # Nhãn dữ liệu
    "labels": ["company", "address", "date", "total", "item", "other"],
}