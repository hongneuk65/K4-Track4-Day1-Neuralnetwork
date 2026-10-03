"""train.py — Huấn luyện, đánh giá và ghi dự đoán CoverType.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import time
import csv
import random
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    parts = [model(X[start:start + batch_size]).argmax(dim=1)
             for start in range(0, len(X), batch_size)]
    return torch.cat(parts).to(dtype=torch.int64)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    loss_sum = 0.0
    cm = np.zeros((7, 7), dtype=np.int64)
    for start in range(0, len(X), batch_size):
        xb, yb = X[start:start + batch_size], y[start:start + batch_size]
        logits = model(xb)
        loss_sum += compute_loss(logits, yb, loss_name, reduction="sum").item()
        pred = logits.argmax(dim=1)
        index = (yb * 7 + pred).to(dtype=torch.int64)
        cm += torch.bincount(index, minlength=49).reshape(7, 7).cpu().numpy()
    return {"loss": loss_sum / len(y), "acc": float(np.trace(cm) / cm.sum()),
            "macro_f1": macro_f1_from_confusion(cm)}


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        # Mean is over all 7 logits; sum is per sample for evaluate().
        errors = (logits - F.one_hot(y, num_classes=7).to(logits.dtype)).square()
        if reduction == "sum":
            return errors.mean(dim=1).sum()
        if reduction == "mean":
            return errors.mean()
    raise ValueError(f"Unknown loss: {loss_name}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg["lr"] is None:
        raise ValueError("Choose lr using validation before running an experiment")
    if cfg["epochs"] < 1 or cfg["batch"] < 1:
        raise ValueError("epochs and batch must be positive")
    set_seed(cfg["seed"])
    device = data["X_tr"].device
    precision = cfg["precision"]
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unknown precision: {precision}")
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("FP16/BF16 lab comparison requires a CUDA device")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support BF16")
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden]
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"],
                                betas=cfg.get("betas", (0.9, 0.999)), eps=cfg.get("eps", 1e-8))
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    step0_loss = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])["loss"]
    keys = ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm", "epoch_time_s")
    history = {key: [] for key in keys}
    best_state, best_epoch, best_val_loss = None, None, float("inf")
    diverged = not np.isfinite(step0_loss)
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Same fixed subset for every run; it is only for the train-loss curve.
    train_count = min(len(data["X_tr"]), 50_000)
    for epoch in range(1, cfg["epochs"] + 1):
        if diverged:
            break
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        grad_total, grad_steps = 0.0, 0
        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], cfg["batch"], generator):
            optimizer.zero_grad(set_to_none=True)
            context = (torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)
                       if precision != "fp32" else nullcontext())
            with context:
                loss = compute_loss(model(xb), yb, cfg["loss"])
            if not torch.isfinite(loss).item():
                diverged = True
                break
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            norm = clip_gradients(model.parameters(), cfg["clip_norm"])
            if not np.isfinite(norm):
                diverged = True
                break
            grad_total += norm
            grad_steps += 1
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        if diverged:
            break
        train_metric = evaluate(model, data["X_tr"][:train_count], data["y_tr"][:train_count], cfg["loss"])
        val_metric = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])
        if not np.isfinite(val_metric["loss"]):
            diverged = True
            break
        values = (epoch, train_metric["loss"], val_metric["loss"], val_metric["acc"],
                  val_metric["macro_f1"], grad_total / max(grad_steps, 1), seconds)
        for key, value in zip(keys, values):
            history[key].append(value)
        if val_metric["loss"] < best_val_loss:
            best_val_loss, best_epoch = val_metric["loss"], epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
    best_index = history["epoch"].index(best_epoch) if best_epoch is not None else None
    summary = dict(step0_loss=step0_loss,
                   best_val_loss=best_val_loss if best_index is not None else None,
                   best_epoch=best_epoch,
                   final_train_loss=history["train_loss"][-1] if best_index is not None else None,
                   final_val_loss=history["val_loss"][-1] if best_index is not None else None,
                   val_acc=history["val_acc"][best_index] if best_index is not None else None,
                   val_macro_f1=history["val_macro_f1"][best_index] if best_index is not None else None,
                   time_per_epoch_s=float(np.mean(history["epoch_time_s"])) if best_index is not None else None,
                   peak_mem_MB=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                   diverged=diverged)
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    ids = np.asarray(row_id)
    labels = np.asarray(preds)
    if ids.shape != labels.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("row_id and predictions must align, with no duplicate row_id")
    if not np.all((0 <= labels) & (labels < 7)):
        raise ValueError("predictions must be in 0..6")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "pred"))
        writer.writerows(zip(ids.tolist(), labels.tolist()))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    if result["best_state"] is None:
        raise ValueError("No best_state is available; check for divergence")
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"])
    model.load_state_dict(result["best_state"])
    model = model.to(data["X_eval"].device)
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
