# Báo cáo Lab Day 1 — Nguyễn Văn Hồng — 2A202602800

## 1. Thiết lập và quy trình

Máy local Windows, Python 3.12.10, PyTorch 2.14.0+cpu. Dữ liệu Forest CoverType gồm 581.012 mẫu, 54 đặc trưng và 7 lớp. Giữ nguyên phép chia metadata: 464.809 train và 116.203 eval. Từ train, tách validation 20% phân tầng với seed 42, còn 371.847 train và 92.962 val. Chỉ 10 đặc trưng số được chuẩn hóa bằng mean/std của **phần train sau khi tách**; 44 đặc trưng nhị phân giữ nguyên. Eval chỉ được dự đoán sau khi đã chốt cấu hình bằng val.

Model M-base tự định nghĩa: `54 → 256 → 128 → 7`, ReLU ở các lớp ẩn, 47.879 tham số, đầu ra là logits. Baseline `base-s1`: He, cross-entropy, SGD + momentum 0,9, learning rate 0,1, batch 512, 20 epoch, FP32, không dropout/clipping. Learning rate 0,1 được chọn bằng val loss sau khi so với `pilot-s1-lr0p03` (`lr=0,03`). Metric chính là macro-F1; đoán luôn lớp 1 cho val accuracy 0,4876 nhưng macro-F1 rất thấp.

Mọi thí nghiệm dùng cùng split và 20 epoch. Train loss được đo ở chế độ `eval()` trên cùng 50.000 mẫu train đầu tiên; val loss trên toàn bộ val. Gradient norm là chuẩn L2 toàn cục **trước** clipping. Mỗi `exp_id` có một dòng trong `experiments.xlsx`, một `results/<exp_id>.json` và một `figures/<exp_id>.png`.

## 2. Kiểm tra ban đầu và độ nhiễu

Model có đúng 47.879 tham số, logits `(B,7)`, mọi tham số nhận gradient khác 0. Loss bước 0 của `base-s1` trên val là **2,2691**, cao hơn `ln(7)=1,9459`: He cho logits ban đầu chưa bằng nhau nên `ln(7)` là mốc tham khảo, không phải giá trị bắt buộc. Phép thử 20 mẫu train đạt accuracy **100%**, loss **0,000986** sau 58 bước Adam ([đường cong](figures/healthcheck_20_samples.png)).

Ba baseline seed `base-s1`, `base-s2`, `base-s3` cho val macro-F1 lần lượt **0,8588**, **0,8447**, **0,8567**; trung bình **0,8534 ± 0,0076** (độ lệch chuẩn mẫu), ngưỡng tham chiếu `2σ = 0,0153`. Val accuracy trung bình **0,9092 ± 0,0003**. Đường [baseline](figures/compare_baseline_seeds.png) cho thấy kết quả F1 dao động giữa seed dù accuracy khá ổn định. Vì chỉ có ba seed, đây là ngưỡng nhiễu thực dụng, không phải kiểm định ý nghĩa thống kê.

## 3. Thí nghiệm trên validation

### Loss

Dự đoán CE sẽ phù hợp hơn MSE vì CE tối ưu trực tiếp xác suất lớp và thường giữ tín hiệu gradient tốt hơn khi dự đoán sai. `loss-mse` đạt val macro-F1 **0,7360**, thấp hơn `base-s1` **0,8588** một khoảng lớn hơn `2σ`; dự đoán được ủng hộ ([ảnh](figures/compare_loss.png)). MSE ở đây là bình phương sai số giữa **logits** và one-hot, lấy trung bình trên mẫu và 7 logit. Val loss MSE **0,0294** và CE **0,2260** khác thang đo, nên không dùng hai số loss để xếp hạng.

### Optimizer và learning rate

Dự đoán Adam/AdamW học tốt ở learning rate nhỏ hơn SGD, còn SGD không momentum hội tụ chậm hơn. Mỗi optimizer đã thử ít nhất hai learning rate; [ảnh chồng](figures/compare_optimizer.png) thể hiện đường val macro-F1. Trong các giá trị đã thử, SGD tốt nhất là `opt-sgd-lr0p1` (**0,7425**); SGD + momentum tốt nhất là `base-s1` (**0,8588**, so với `pilot-s1-lr0p03` **0,8253**); Adam `opt-adam-lr0p003` và AdamW `opt-adamw-lr0p003` cùng **0,8698** (cặp lr 0,001 của cả hai là **0,8441**). Adam/AdamW với weight decay **0** cho kết quả trùng nhau như dự đoán; thí nghiệm này chưa kiểm tra lợi ích của weight decay tách riêng. Mức tăng **0,0110** của Adam so với baseline seed 1 nhỏ hơn `2σ=0,0153`, nên chưa đủ bằng chứng để khẳng định Adam thắng vững chắc.

### Batch size và dropout

Dự đoán batch lớn có ít bước cập nhật hơn ở cùng 20 epoch. `batch-2048` đạt **0,8075**, thấp hơn `base-s1` **0,8588**; thời gian huấn luyện trung bình mỗi epoch giảm từ **1,96 s** xuống **1,33 s**, nhưng số bước cập nhật giảm khoảng bốn lần ([ảnh](figures/compare_hparam.png)). Đây là so sánh cùng epoch chứ không cùng số bước; cần thử thêm learning rate thích hợp cho batch lớn trước khi kết luận chung về batch size.

Dự đoán dropout 0,3 có thể hại khi baseline chưa quá khớp mạnh. `drop-0p3` đạt **0,7698**, thấp hơn baseline **0,0890** ([ảnh](figures/compare_dropout.png)). Khoảng val–train loss tại best epoch giảm từ khoảng **0,0244** xuống **0,0039**, nhưng cả train/val loss đều cao hơn. Điều này phù hợp với chính quy hóa quá mạnh trong cấu hình hiện tại; dropout không tự động cải thiện mô hình.

### Gradient clipping

Grad norm trung bình của `base-s1` quanh **0,58**, nên ngưỡng **0,3** thấp hơn nhiều bước và thật sự cắt gradient. `clip-0p3` ở learning rate 0,1 đạt **0,8225**, thấp hơn baseline **0,8588**: cắt quá chặt đã hạn chế cập nhật. Ở learning rate 0,5, cặp `highlr-0p5` / `highlr-0p5-clip-0p3` đạt **0,8429 / 0,8461** ([ảnh](figures/compare_clipping.png)). Lượt không clip không NaN; chênh **0,0032** dưới nhiễu. Do đó dữ liệu này **chưa chứng minh** clipping cứu một trường hợp mất ổn định. Hai lượt lr cao chỉ được so trực tiếp với nhau vì cả hai đã đổi lr so với baseline.

### Khởi tạo tham số

Dự đoán zeros không phá được đối xứng giữa các nơ-ron và làm ReLU ở 0 bất hoạt. `init-zeros` đạt accuracy **0,4876**, macro-F1 **0,0936**, gần như chỉ đoán lớp đa số; trái lại `init-xavier` đạt **0,8418** và He baseline **0,8588** ([ảnh](figures/compare_init.png)). Chênh Xavier–He **0,0171** chỉ nhỉnh hơn ngưỡng `2σ`, nên cần lặp thêm seed. Độ lệch chuẩn kích hoạt sau ba lớp Linear ở bước 0: He **[0,6662; 0,6464; 0,5933]**, Xavier **[0,2781; 0,2203; 0,1969]**, zeros **[0; 0; 0]**. He dùng phương sai trọng số `2/n_in` cho ReLU; Xavier dùng `2/(n_in+n_out)` và cho kích hoạt nhỏ hơn trong phép đo này. Mạng chỉ có hai lớp ẩn, nên chưa đủ để kết luận về hiện tượng suy giảm ở mạng rất sâu.

### Mixed precision

Không chạy so FP32/FP16/BF16: PyTorch cài trên máy là bản CPU, không có CUDA cho quy trình AMP của GUIDE. Vì vậy báo cáo **không kết luận** mixed precision nhanh hơn hay tiết kiệm bộ nhớ trên GPU. Đây là một chủ đề chưa được phủ trong thí nghiệm.

## 4. Đánh giá cuối trên eval

Trước khi mở eval, chọn `opt-adam-lr0p003` bằng **best val loss 0,2128** (thấp nhất trong các cấu hình đã thử; AdamW lr 0,003 bằng điểm nhưng weight decay 0). Giữ M-base, He, CE, batch 512, FP32, seed 1 và trọng số **epoch 18**. Baseline `base-s1` có best val loss **0,2260**, epoch 20. Sau khi chọn, huấn luyện lại đúng seed để lấy trọng số trong RAM; val loss và best epoch tái lập, rồi mới dự đoán eval. Các số sau do `scripts/evaluate.py` sinh ra, lưu trong `eval_result.json` và `results/eval/base-s1_eval.json`.

| Cấu hình | Val macro-F1 | Eval accuracy | Eval macro-F1 |
|---|---:|---:|---:|
| `base-s1` | 0,8588 | 0,9086 | 0,8586 |
| `opt-adam-lr0p003` | 0,8698 | 0,9136 | 0,8698 |

Eval macro-F1 tăng **0,0112**, nằm trong mức nhiễu tham chiếu `2σ=0,0153` đo trên validation. Đây là kết quả của một seed trên eval; chưa thể khẳng định cải thiện vượt nhiễu seed. Val và eval macro-F1 gần nhau cho cả hai cấu hình. Không điều chỉnh cấu hình sau khi thấy điểm eval.

### Phân tích lỗi theo lớp của cấu hình cuối

| Lớp | Support | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0 | 42.368 | 0,9029 | 0,9216 | 0,9122 |
| 1 | 56.661 | 0,9333 | 0,9171 | 0,9251 |
| 2 | 7.151 | 0,8954 | 0,9263 | 0,9106 |
| 3 | 549 | 0,8980 | 0,7377 | 0,8100 |
| 4 | 1.899 | 0,7732 | 0,7936 | 0,7833 |
| 5 | 3.473 | 0,8547 | 0,8013 | 0,8272 |
| 6 | 4.102 | 0,9074 | 0,9342 | 0,9206 |

Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán; cùng số trong `eval_result.json`):

```text
          0      1     2    3     4     5     6
0     39048   2937     0    0    48     8   327
1      3929  51961   144    0   379   184    64
2         0    246  6624   34    13   234     0
3         0      3    99  405     0    42     0
4        31    335    21    0  1507     5     0
5         9    157   510   12     2  2783     0
6       232     38     0    0     0     0  3832
```

Lớp khó nhất theo F1 là **lớp 4 (Aspen), 0,7833**; 335/1.899 mẫu lớp này bị dự đoán thành lớp 1. Lớp 3 chỉ có 549 mẫu, recall 0,7377 và bị nhầm với lớp 2 ở 99 mẫu. Mất cân bằng lớp là một nguyên nhân hợp lý; chưa phân tích từng đặc trưng của các mẫu nhầm nên không khẳng định cụ thể đặc điểm địa hình gây nhầm. Bước tiếp theo có thể kiểm tra phân bố đặc trưng và thử loss có trọng số lớp, nhưng phải chọn lại bằng validation.

## 5. Trả lời câu hỏi dẫn dắt và hạn chế

1. **Optimizer:** ở các lr đã thử, Adam/AdamW đạt val macro-F1 cao nhất (0,8698), nhưng hơn baseline dưới `2σ`; SGD không momentum thấp hơn nhiều. Nếu ép mọi optimizer dùng cùng lr 0,1 thì không công bằng vì Adam cần thang lr khác. AdamW chưa được thử với weight decay dương.
2. **Dropout:** không giúp trong cấu hình này; có thể hữu ích khi train loss tiếp tục giảm còn val loss tăng rõ và lặp lại qua seed. Kết quả `drop-0p3` giảm F1 phản ánh dropout 0,3 quá mạnh, không chứng minh mọi mức dropout đều hại.
3. **Clipping:** giới hạn độ dài bước cập nhật khi gradient tăng đột ngột. Ở đây clip 0,3 thực sự tác động nhưng không có lượt không clip bị mất ổn định, nên chưa có minh chứng nó giải quyết lỗi NaN; cặp lr cao chênh nhỏ hơn nhiễu.
4. **Mixed precision:** chưa đo trên GPU, nên không kết luận về tốc độ hoặc bộ nhớ. Mạng nhỏ có thể bị chi phí khởi động kernel chi phối, nhưng đó là giả thuyết chưa được kiểm chứng ở máy này.
5. **Khởi tạo:** zeros giữ các nơ-ron ẩn giống nhau và ReLU(0) làm gradient không lan qua tầng ẩn, dẫn đến dự đoán gần lớp đa số. He giữ phương sai phù hợp hơn với ReLU so với Xavier trong phép đo kích hoạt; để đánh giá chắc khác biệt chất lượng cần nhiều seed hơn.
6. **Nếu loss không giảm sau 2.000 bước:** (i) kiểm tra nhãn `0..6`, shape/dtype, chuẩn hóa chỉ từ train và loss bước 0; lỗi ở đây chỉ ra dữ liệu/tiền xử lý; (ii) tắt dropout và thử quá khớp 20 mẫu, kiểm tra logits và gradient từng lớp; không làm được gợi ý lỗi model, loss hoặc luồng gradient; (iii) kiểm tra `zero_grad → forward → loss → backward → optimizer.step`, learning rate, grad norm và NaN theo bước; phép thử này phân biệt lỗi vòng lặp với lr quá nhỏ/quá lớn. Thứ tự này rẻ hơn chạy thêm 2.000 bước mù.

Hạn chế chính: hầu hết biến thể chỉ một seed, optimizer chỉ có hai lr mỗi loại, thí nghiệm batch giữ cùng epoch nhưng không cùng bước cập nhật, chưa thử AMP trên GPU. Best epoch chọn bằng **val loss**, trong khi metric chấm là macro-F1; một quy tắc chọn khác có thể cho kết quả khác và phải xác định trước khi xem eval. Nếu có thêm thời gian, nên lặp seed cho các ứng viên mạnh nhất, thử clipping trong tình huống thật sự mất ổn định, thử weight decay dương và phân tích lỗi theo đặc trưng.

## 6. File nộp

`code/lab.ipynb` và các module `.py`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, `figures/`, `results/`. Không nộp dữ liệu, checkpoint hay cache. Tổng thời gian **phần huấn luyện trong các thí nghiệm đã lưu** khoảng **798 giây** trên CPU; con số này không gồm nạp dữ liệu, vẽ ảnh, đánh giá và hai lượt huấn luyện lại trước eval.
