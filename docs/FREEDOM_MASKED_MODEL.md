# FREEDOM_MASKED: shared item, separate user và CL giữa hai branch

## 1. Mô hình làm gì?

FREEDOM_MASKED dự đoán mức độ quan tâm của user đối với item bằng cách kết hợp hai U–I view và một multimodal I–I graph.

- **Original branch** giữ U–I propagation của FREEDOM: degree-sensitive edge dropout khi train, full train graph khi inference.
- **Masked branch** học trọng số trên các cạnh tương tác để điều chỉnh thông tin được truyền trong graph.
- **Symmetric CL** căn chỉnh representation của cùng một user/item giữa hai U–I branch, với gradient cập nhật cả hai phía.
- **Gated-sum** kết hợp hai U–I output.
- **Parallel I–I branch** truyền shared item-ID embedding qua frozen multimodal item graph, rồi cộng vào fused U–I item output.

Tài liệu mô tả cấu trúc lõi trong `src/models/freedom_masked.py`, với **item-ID table dùng chung và user-ID table riêng**. Không bao gồm modality auxiliary BPR, branch auxiliary BPR, relation loss, teacher CL, dropout CL hay các item-input/fusion mode khác.

## 2. Phạm vi cấu hình

```yaml
embedding_size: 64
feat_embed_dim: 64
n_ui_layers: 2
n_mm_layers: 1
knn_k: 10
mm_image_weight: 0.1

user_embedding_mode: separate
item_embedding_mode: shared
item_input_mode: id
ui_branch_mode: dual
ui_fusion_mode: gated_sum
ui_gate_mode: separate
gate_init_mode: xavier

dropout: 0.8
mask_graph_mode: soft
mask_keep_ratio: 0.3
mask_degree_mode: full
mask_weight: 0.1
mask_binary_weight: 0.1

cl_mode: symmetric
cl_weight: 0.05
cl_temperature: 0.2

# Tắt các loss nằm ngoài cấu trúc cơ bản này.
reg_weight: 0.0
aux_bpr_mode: none
mask_relation_mode: none
```

Đây là cấu hình minh họa, không phải hyperparameter được đảm bảo tốt nhất. Tài liệu không thay đổi code hay config hiện tại.

## 3. Embedding đầu vào

Gọi dimension của ID embedding là \(d\), ví dụ \(d=64\).

Hai user table độc lập:

\[
E_u^f\in\mathbb R^{N_u\times d},
\qquad
E_u^m\in\mathbb R^{N_u\times d}.
\]

Một item-ID table dùng chung:

\[
E_i\in\mathbb R^{N_i\times d}.
\]

Các table được khởi tạo bằng Xavier và học bằng gradient. Hai user table được khởi tạo độc lập; không được copy-initialize trong code hiện tại.

Input của hai U–I branch:

\[
H_f^{(0)}=[E_u^f;E_i],
\qquad
H_m^{(0)}=[E_u^m;E_i].
\]

\([\cdot;\cdot]\) là ghép theo hàng: user trước, item sau. **Không concatenate image/text feature vào item input**, và không tự tăng dimension thành \(2d\).

Shared item input không đồng nghĩa shared item output: hai graph và hai user table khác nhau vẫn tạo ra item representation khác nhau.

## 4. Hai U–I graph

### Original FREEDOM branch

Gọi \(A\) là bipartite adjacency được xây dựng chỉ từ interaction trong tập train:

\[
A=\begin{bmatrix}0&R\\R^\top&0\end{bmatrix},
\qquad
\widehat A_f=D^{-1/2}AD^{-1/2}.
\]

Khi train, FREEDOM resample interaction bằng degree-sensitive sampling mỗi epoch. Với `dropout: 0.8`, graph giữ khoảng 20% interaction. Mỗi interaction được giữ ở cả hai chiều và sampled graph được degree-normalize lại.

Ký hiệu adjacency thực sự dùng bởi original branch là:

\[
\widehat A_o=
\begin{cases}
\widehat A_{drop},&\text{train},\\
\widehat A_f,&\text{inference}.
\end{cases}
\]

Trong code, output vẫn mang tên `full_users`/`full_items`, nhưng khi train chúng là output của **degree-dropout graph**, không nhất thiết của full graph.

### Learned masked branch

Với mỗi train interaction \((u,i)\), model học một scalar logit:

\[
m_{ui}=\sigma(\theta_{ui})\in(0,1).
\]

Mask dùng chung cho hai chiều cạnh. Với `soft` và `mask_degree_mode: full`:

\[
\widehat A_m=\widehat A_f\odot M,
\qquad
(\widehat A_m)_{ui}=\frac{m_{ui}}{\sqrt{d_ud_i}}.
\]

Masked branch không nhận degree-dropout của original branch. Ban đầu mọi mask bằng \(\rho=\texttt{mask\_keep\_ratio}\), do logits được khởi tạo bằng \(\operatorname{logit}(\rho)\).

Soft mask điều chỉnh trọng số chứ không xóa cạnh; \(\rho\) là mục tiêu trung bình mask, không phải sparsity được đảm bảo.

## 5. U–I propagation và fusion

Hai branch cùng dùng LightGCN-style propagation:

\[
H_f^{(\ell+1)}=\widehat A_oH_f^{(\ell)},
\qquad
H_m^{(\ell+1)}=\widehat A_mH_m^{(\ell)}.
\]

Output là trung bình layer 0 và các GCN layer:

\[
H_f=\frac1{L_{UI}+1}\sum_{\ell=0}^{L_{UI}}H_f^{(\ell)},
\qquad
H_m=\frac1{L_{UI}+1}\sum_{\ell=0}^{L_{UI}}H_m^{(\ell)}.
\]

Tách theo hàng để thu được user/item U–I output \(U_f,I_f\) và \(U_m,I_m\).

Với `ui_gate_mode: separate`, user và item dùng hai gate network riêng:

\[
g_u=\sigma(W_u[U_f\Vert U_m]+b_u),
\qquad
g_i=\sigma(W_i[I_f\Vert I_m]+b_i).
\]

\[
U_{UI}=g_u\odot U_f+(1-g_u)\odot U_m,
\]

\[
I_{UI}=g_i\odot I_f+(1-g_i)\odot I_m.
\]

Gate được áp dụng độc lập theo từng node và từng dimension. User không bị concatenate với item. Gated-sum giữ nguyên dimension \(d\).

Với Xavier initialization, gate ban đầu phụ thuộc feature và thường quanh 0.5; `gate_initial_original_weight` không điều khiển gate trong mode này.

## 6. Parallel frozen I–I graph

Image/text feature đã trích xuất sẵn được dùng để xây dựng normalized KNN item graph:

\[
A_{MM}=\alpha A_v+(1-\alpha)A_t,
\qquad \alpha=\texttt{mm\_image\_weight}.
\]

I–I adjacency được xây dựng/load từ cache và giữ cố định. Nó truyền **shared item-ID embedding**, không truyền image/text feature hay item output sau U–I:

\[
H_{MM}=A_{MM}^{L_{MM}}E_i.
\]

Vì item table dùng chung, hai branch có cùng I–I output. Code chỉ cần tính một lần, không cần MM fusion gate:

\[
g\odot H_{MM}+(1-g)\odot H_{MM}=H_{MM}.
\]

Representation cuối:

\[
U_{final}=U_{UI},
\qquad
I_{final}=I_{UI}+H_{MM}.
\]

```text
Original user-ID ─┐
Shared item-ID ───── Degree-dropout U–I ── U_f, I_f ─┐
                                                    ├─ gated-sum ─ U_final
Masked user-ID ───┐                                 │              I_UI ─┐
Shared item-ID ───── Learned masked U–I ── U_m, I_m ─┘                    ├─ I_final
                                                                        │
Shared item-ID ───── Frozen multimodal I–I graph ── H_MM ────────────────┘

          U_f ↔ U_m và I_f ↔ I_m: symmetric CL trước fusion
```

U–I và I–I chạy song song, không có đường truyền `U–I output → I–I input`.

## 7. Symmetric CL giữa hai U–I branch

Trong mỗi training batch, lấy unique user \(\mathcal B_u\) và unique positive item \(\mathcal B_i\). Negative item sample cho BPR không tự động được thêm vào item CL.

Với hai view \(X,Y\) của cùng tập node \(\mathcal B\), cosine similarity và InfoNCE một chiều là:

\[
s(x,y)=\frac{x^\top y}{\|x\|_2\|y\|_2},
\]

\[
\ell(X\rightarrow Y)
=-\frac1{|\mathcal B|}\sum_{a\in\mathcal B}
\log\frac{\exp(s(x_a,y_a)/\tau)}
{\sum_{b\in\mathcal B}\exp(s(x_a,y_b)/\tau)}.
\]

\[
\operatorname{InfoNCE}_{sym}(X,Y)
=\frac12[\ell(X\rightarrow Y)+\ell(Y\rightarrow X)].
\]

\[
L_{CL}=\frac12\left[
\operatorname{InfoNCE}_{sym}(U_f[\mathcal B_u],U_m[\mathcal B_u])
+\operatorname{InfoNCE}_{sym}(I_f[\mathcal B_i],I_m[\mathcal B_i])
\right].
\]

Cùng user/item giữa hai branch là positive pair; node khác trong tập CL là negative. Không đối chiếu user với item.

Với `cl_mode: symmetric`, CL dùng original **degree-dropout U–I view khi train** và masked U–I view. Không dùng full-graph teacher bổ sung hay `stopgrad`.

CL nhận U–I output trước fusion và trước I–I residual. Gradient cập nhật hai user table, shared item table và learned mask qua masked propagation; không trực tiếp đi qua fusion gate hoặc I–I propagation.

## 8. Loss cơ bản

Điểm recommendation và BPR:

\[
\widehat y_{ui}=u_{final,u}^\top i_{final,i},
\]

\[
L_{BPR}=-\frac1{|\mathcal T|}\sum_{(u,i^+,i^-)\in\mathcal T}
\log\sigma(\widehat y_{ui^+}-\widehat y_{ui^-}).
\]

Mask regularization trên tất cả train edge:

\[
L_{budget}=\left(\frac1{|\mathcal E|}\sum_{e\in\mathcal E}m_e-\rho\right)^2,
\]

\[
L_{binary}=\frac1{|\mathcal E|}\sum_{e\in\mathcal E}m_e(1-m_e),
\qquad
L_{mask}=L_{budget}+\beta L_{binary}.
\]

Loss tổng của cấu hình này:

\[
\boxed{L=L_{BPR}+\lambda_{CL}L_{CL}+\lambda_{mask}L_{mask}}
\]

\(\lambda_{CL}=\texttt{cl\_weight}\), \(\lambda_{mask}=\texttt{mask\_weight}\), \(\beta=\texttt{mask\_binary\_weight}\).

- BPR học ranking, fusion và ID embedding qua cả U–I/I–I propagation.
- CL căn chỉnh hai U–I view.
- Mask regularization khuyến khích trung bình mask gần budget và giá trị gần 0/1.

Với `item_input_mode: id` và `reg_weight: 0`, modality embedding/projection không nhận loss auxiliary. Multimodal feature vẫn có tác dụng qua frozen I–I graph đã xây dựng, nhưng không trực tiếp làm U–I input.

## 9. Inference

Khi `model.eval()`:

- Original branch dùng full **train** graph, không degree-dropout.
- Masked branch dùng learned soft-mask graph.
- Gate kết hợp hai U–I view, rồi item output nhận shared I–I residual.
- Xếp hạng bằng dot product; không tính CL/loss.

Validation/test interaction không được thêm vào graph. Không cần đổi YAML `dropout` về 0 để inference.

## 10. Khác biệt chính so với PGL_MASKED cơ bản

| Thành phần | PGL_MASKED | FREEDOM_MASKED trong tài liệu này |
|---|---|---|
| Item input | Concatenate projected image/text | Shared item-ID embedding |
| User input | Concatenate hai user table theo modality, mỗi branch riêng | Một user-ID table cho mỗi branch |
| Dimension | \(2d\) với cấu hình đồng kích thước | \(d\) |
| Original U–I lúc train | Full graph | Degree-sensitive dropout graph |
| I–I propagation input | Multimodal item embedding | Shared item-ID embedding |
| Branch CL | Symmetric, trước fusion | Symmetric, trước fusion |

Mask và CL có vai trò học một view bổ sung, không đảm bảo luôn vượt FREEDOM gốc. Đặc biệt, original view thay đổi theo degree-dropout và shared item table nhận gradient từ cả hai branch; lợi ích thực tế cần được kiểm chứng bằng ablation và nhiều seed.
