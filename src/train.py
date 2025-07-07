import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np
import os

from src.model import InvoiceGCN


def run_training_session(config, train_data, test_data, save_path):

    # 1. KHỞI TẠO
    # ==================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    config = config['model_params']
    # Khởi tạo mô hình với các siêu tham số từ config
    model = InvoiceGCN(
        input_dim=train_data.x.shape[1],
        hidden_dims=config['hidden_dims'],
        n_classes=config['n_classes'],
        dropout_rate=config['dropout_rate'],
        chebnet=config.get('chebnet', True), # .get() để có giá trị mặc định
        K=config.get('K', 3)
    ).to(device)

    # Khởi tạo optimizer với các siêu tham số từ config
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    train_data = train_data.to(device)
    test_data = test_data.to(device)


    # 2. XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU
    # ==================================
    _class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_data.y.cpu().numpy()),
        y=train_data.y.cpu().numpy()
    )
    class_weights_tensor = torch.tensor(_class_weights, dtype=torch.float).to(device)
    print(f"Trọng số của các lớp: {_class_weights}")


    # 3. VÒNG LẶP HUẤN LUYỆN
    # ==================================
    num_epochs = config['num_epochs']
    print(f"\nBắt đầu quá trình huấn luyện với {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(train_data)

        # Đảm bảo nhãn bắt đầu từ 0 cho hàm loss
        y_true_train = train_data.y - 1
        loss = F.nll_loss(out, y_true_train, weight=class_weights_tensor)

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                # Đảm bảo nhãn bắt đầu từ 0 cho hàm loss
                y_true_test = test_data.y - 1
                val_out = model(test_data)
                val_loss = F.nll_loss(val_out, y_true_test)

                print("-" * 50)
                print(f"Epoch: {epoch:04d}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

                for name in ['train', 'test']:
                    _data = train_data if name == 'train' else test_data
                    y_pred = model(_data).max(dim=1)[1]
                    y_true = (_data.y - 1)
                    acc = y_pred.eq(y_true).sum().item() / _data.num_nodes
                    print(f"\t{name.capitalize()} Accuracy: {acc:.4f}")

                    if name == 'test':
                        y_pred_np = y_pred.cpu().numpy()
                        y_true_np = y_true.cpu().numpy()
                        print("\nBáo cáo chi tiết trên tập Test:")
                        # Thêm target_names để báo cáo dễ đọc hơn
                        target_names = ['company', 'address', 'date', 'total', 'item', 'undefined'][:config['n_classes']]
                        print(classification_report(y_true_np, y_pred_np, target_names=target_names, zero_division=0))
                print("-" * 50)

    print("\nHoàn tất huấn luyện!")

    # 4. LƯU MÔ HÌNH ĐÃ HUẤN LUYỆN
    # ==================================
    # ĐẢM BẢO RẰNG BẠN ĐANG LẤY ĐƯỜNG DẪN TỪ CONFIG:
    save_path = config['model_save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Đã lưu mô hình tại: {save_path}")

    torch.save(model.state_dict(), save_path)
    print(f"\nĐã lưu mô hình đã huấn luyện tại: {save_path}")