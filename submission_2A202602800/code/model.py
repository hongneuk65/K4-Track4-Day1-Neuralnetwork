"""model.py — MLP được định nghĩa theo kiến trúc cố định của bài lab.

Model: MLP cho bài toán 7 lớp, shape cố định (xem README mục 3 và GUIDE, "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS bên dưới.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        hidden = tuple(hidden)
        if hidden not in EXPECTED_PARAMS or in_features != 54 or num_classes != 7:
            raise ValueError("Use one of the lab's fixed 54-input, 7-class architectures")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        layers = []
        previous = in_features
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.net = nn.Sequential(*layers)
        init_weights(self, init)
        assert count_params(self) == EXPECTED_PARAMS[hidden]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo mọi nn.Linear; bias = 0 trừ chế độ default của PyTorch.

    init:
        "zeros"   : W = 0
        "normal"  : W ~ N(0, 0.01^2)
        "xavier"  : nn.init.xavier_normal_ (Var = 2/(n_in+n_out)); nếu bạn dùng Var = 1/n_in theo slide, hãy ghi rõ
        "he"      : nn.init.kaiming_normal_(w, nonlinearity="relu")  (Var = 2/n_in)
        "default" : không làm gì (giữ khởi tạo mặc định của nn.Linear; KHÔNG phải He)
    Gợi ý: duyệt model.modules(), chọn isinstance(m, nn.Linear).
    """
    if init not in {"zeros", "normal", "xavier", "he", "default"}:
        raise ValueError(f"Unknown initialization: {init}")
    if init == "default":
        return
    for layer in model.modules():
        if not isinstance(layer, nn.Linear):
            continue
        if init == "zeros":
            nn.init.zeros_(layer.weight)
        elif init == "normal":
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        elif init == "xavier":
            nn.init.xavier_normal_(layer.weight)
        else:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được. Dùng để assert với EXPECTED_PARAMS ngay sau khi tạo model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt sau mỗi lớp (ở bước 0, một lô val) — dùng cho thí nghiệm khởi tạo.

    Các bước:
      1. model.eval(); h = x
      2. duyệt từng lớp con theo thứ tự; sau mỗi nn.Linear (hoặc sau mỗi ReLU, bạn chọn và ghi rõ) lưu h.std().item()
      3. trả về danh sách std theo lớp
    """
    model.eval()
    values = []
    h = x
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.Linear):
            values.append(float(h.std().item()))
    return values
