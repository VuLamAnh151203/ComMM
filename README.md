# ComMM

`ComMM` là bản tách gọn của các mô hình PGL và FREEDOM trong workspace này. Thư
mục chỉ giữ bốn model cần thiết cùng hạ tầng huấn luyện/đánh giá dùng chung:

- `PGL`: PGL gốc.
- `PGL_MASKED`: PGL với nhánh user-item graph được mask.
- `FREEDOM`: FREEDOM gốc, gồm degree-sensitive U-I dropout và frozen multimodal
  I-I graph.
- `FREEDOM_MASKED`: FREEDOM với một nhánh U-I dropout độc lập và một nhánh mask;
  model này không dùng contrastive learning.

Hai model masked dùng sparse backward chỉ trên các cạnh quan sát, tránh tạo
gradient adjacency đặc có kích thước `(n_users + n_items)^2`.

## Cài đặt

Khuyến nghị Python 3.10+:

```bash
python -m pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

Dữ liệu không được copy vào project. Nếu không truyền `--data-path`, chương
trình tìm dataset tại `ComMM/data/<dataset>/`. Có thể đặt dữ liệu ở nơi khác và
truyền thư mục cha bằng `--data-path`.

Ví dụ dataset `baby`:

```text
data/baby/
|-- baby.inter
|-- image_feat.npy
`-- text_feat.npy
```

File `.inter` có ba cột tab-separated `userID`, `itemID`, `x_label`, trong đó
`x_label` lần lượt là `0` (train), `1` (validation), `2` (test). ID user/item là
số nguyên liên tiếp bắt đầu từ 0. Mỗi file feature `.npy` có một hàng cho mỗi
item, theo đúng thứ tự item ID. FREEDOM hỗ trợ một hoặc cả hai modality; PGL yêu
cầu cả image và text feature.

## Chạy

Từ thư mục `ComMM`:

```bash
python src/main.py --model PGL --dataset baby
python src/main.py --model PGL_MASKED --dataset baby
python src/main.py --model FREEDOM --dataset baby
python src/main.py --model FREEDOM_MASKED --dataset baby
```

Hoặc có thể `cd ComMM/src` rồi chạy trực tiếp:

```bash
python main.py -m FREEDOM_MASKED -d baby
```

Ví dụ dùng dữ liệu bên ngoài và CPU:

```bash
python src/main.py -m FREEDOM_MASKED -d movielens_1m \
  --data-path C:/datasets/mmrec --cpu
```

`FREEDOM_MASKED.yaml` mặc định dùng hard mask, hai bộ user/item-ID embedding
riêng và `gated_concat` giữ nguyên output `2d`. Các mode mask được hỗ trợ là
`soft`, `hard`, `double_full`, `svd`, `local_prunning`, `random_fixed` và
`random_dynamic`. `random_fixed` giữ một random mask xuyên suốt; `random_dynamic`
sample lại khi bắt đầu mỗi epoch train nhưng dùng một evaluation mask cố định
theo `random_mask_seed`. Checkpoint, artifact mask và top-k recommendation được
ghi theo các đường dẫn trong config.

Implementation `FREEDOM_MASKED` cũ được giữ tại
`src/models/freedom_masked_legacy.py`. Implementation mới giữ residual của
FREEDOM (`I_UI + I_MM`), hỗ trợ `ui_gate_mode: shared|separate`,
`mm_gate_mode: reuse_ui_item|separate`, và contrastive loss giữa hai U-I view.
`item_input_mode: id|multimodal|multimodal_concat|hybrid` cho phép giữ item-ID
input gốc, project multimodal input về `d`, hoặc giữ `[image d | text d]` ở
dimension `2d` trong cả U-I và I-I graph.
Gate hỗ trợ `gate_init_mode: xavier|constant`; mode `constant` khởi tạo tỷ lệ
nhánh gốc bằng `gate_initial_original_weight` trước khi MLP tiếp tục học.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
```

Các test dùng dữ liệu tổng hợp tạm thời, không cần dataset thật.

## Nguồn

Phần PGL dựa trên implementation cho bài báo AAAI 2025 *Mind Individual
Information! Principal Graph Learning for Multimedia Recommendation*. FREEDOM
dựa trên implementation *A Tale of Two Graphs: Freezing and Denoising Graph
Structures for Multimodal Recommendation* trong `MMRec`. Xem giấy phép và thông
báo sử dụng học thuật của các repository nguồn trong workspace.
