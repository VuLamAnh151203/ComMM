# PGL kết hợp learnable mask và contrastive learning giữa hai branch

## 1. Mô hình làm gì?

Mô hình dự đoán item mà một user có khả năng quan tâm bằng cách kết hợp thông tin tương tác user–item và đặc trưng image/text của item.

Hai U–I branch cùng học từ lịch sử tương tác:

- **Full branch** sử dụng toàn bộ cạnh tương tác trong tập train.
- **Masked branch** học trọng số cho từng cạnh, nhằm điều chỉnh ảnh hưởng của các tương tác lên embedding.

Contrastive learning (CL) căn chỉnh representation của cùng một user/item giữa hai branch. Một learned gate kết hợp hai view để tính điểm recommendation. Item representation còn nhận thông tin từ một multimodal I–I graph chạy song song với U–I graph.

Đây là mô tả cấu trúc lõi trong `src/models/pgl_masked.py`, không bao gồm dropout CL, teacher/stop-gradient CL, random mask, SVD, local pruning hay các loss phụ của model khác. Tên PGL bên dưới chỉ nền tảng kiến trúc; không có nghĩa tài liệu mô tả đầy đủ mọi thành phần của PGL gốc.

## 2. Phạm vi cấu hình

Để các công thức nhất quán, xét cấu hình:

```yaml
embedding_size: 64
feat_embed_dim: 64
user_embedding_mode: separate
ui_branch_mode: dual
ui_fusion_mode: gated_sum
mask_graph_mode: soft
mask_degree_mode: full
cl_mode: branch
cl_weight: 0.1
cl_temperature: 0.2
mask_keep_ratio: 0.3
mask_weight: 0.1
mask_binary_weight: 0.1
```

Đây là cấu hình minh họa, không phải cam kết về hyperparameter tốt nhất và không thay đổi config đang dùng của project.

Gọi kích thước mỗi modality là \(d\). Embedding đưa vào graph có kích thước \(q=2d\); với ví dụ trên, \(q=128\).

## 3. Embedding đầu vào

### Item: dùng modality feature, không dùng item-ID embedding

Mỗi item \(i\) có feature image \(x_i^v\) và text \(x_i^t\) đã được trích xuất trước và lưu trong các file `.npy`. Model không encode trực tiếp ảnh hoặc text thô.

Hai feature matrix được load thành embedding table có thể học (`freeze=False`), rồi project và L2-normalize:

\[
z_i^v=\operatorname{Normalize}(W_vx_i^v+b_v),
\qquad
z_i^t=\operatorname{Normalize}(W_tx_i^t+b_t).
\]

Item input là concatenate hai modality:

\[
e_i=[z_i^v\Vert z_i^t]\in\mathbb{R}^{q}.
\]

Hai U–I branch **dùng chung item input** \(e_i\), chung modality embedding table và chung projection. Item output vẫn khác nhau vì được truyền qua hai graph khác nhau.

### User: embedding table riêng cho mỗi branch

Mỗi branch có hai user-ID embedding table, tương ứng với hai phần image/text:

\[
e_u^f=[e_{u,v}^f\Vert e_{u,t}^f],
\qquad
e_u^m=[e_{u,v}^m\Vert e_{u,t}^m].
\]

Ký hiệu \(f\) là full branch và \(m\) là masked branch. Các user table được khởi tạo độc lập và học bằng gradient; tên image/text chỉ hai phần embedding, không có user image/text feature đầu vào.

## 4. Hai U–I graph

### Full graph

Gọi \(R\) là interaction matrix của **tập train**. Validation/test interaction không được thêm vào graph.

\[
A=
\begin{bmatrix}
0 & R\\
R^\top & 0
\end{bmatrix},
\qquad
\widehat A_f=D^{-1/2}AD^{-1/2}.
\]

\(D\) là degree matrix của full train graph. Full branch luôn dùng graph này, không có degree-dropout trong cấu trúc đang mô tả.

### Learnable soft mask

Với mỗi cạnh train \((u,i)\), model học một scalar logit \(\theta_{ui}\):

\[
m_{ui}=\sigma(\theta_{ui})\in(0,1).
\]

Hai chiều \(u\rightarrow i\) và \(i\rightarrow u\) dùng chung mask. Mask chỉ được đặt trên cạnh đã quan sát, không tạo thêm interaction mới.

Với `mask_degree_mode: full`, giữ nguyên degree normalization của full graph:

\[
\widehat A_m=\widehat A_f\odot M,
\qquad
(\widehat A_m)_{ui}
=\frac{m_{ui}}{\sqrt{d_ud_i}}.
\]

Ban đầu, \(\theta_{ui}=\operatorname{logit}(\rho)\), với \(\rho=\texttt{mask\_keep\_ratio}\), nên mọi mask bằng \(\rho\).

**Soft mask không thực sự xóa cạnh.** Nó giảm/tăng trọng số cạnh; \(\rho\) là mục tiêu trung bình mask, không phải tỷ lệ cạnh được đảm bảo giữ lại.

## 5. U–I graph propagation

Embedding ban đầu của hai branch:

\[
H_f^{(0)}=[E_u^f;E_i],
\qquad
H_m^{(0)}=[E_u^m;E_i].
\]

\([\cdot;\cdot]\) ở đây là ghép theo hàng: user ở phía trên và item ở phía dưới; khác với concatenate feature \(\Vert\).

Tại mỗi layer:

\[
H_f^{(\ell+1)}=\widehat A_fH_f^{(\ell)},
\qquad
H_m^{(\ell+1)}=\widehat A_mH_m^{(\ell)}.
\]

Không có nonlinear activation giữa các U–I layer. Output là trung bình layer 0 và các layer GCN:

\[
H_f=\frac{1}{L_{UI}+1}\sum_{\ell=0}^{L_{UI}}H_f^{(\ell)},
\qquad
H_m=\frac{1}{L_{UI}+1}\sum_{\ell=0}^{L_{UI}}H_m^{(\ell)}.
\]

Tách theo hàng để thu được \(U_f,I_f\) và \(U_m,I_m\). Đây là representation U–I **trước fusion và trước khi cộng I–I residual**.

## 6. Gated-sum fusion

Với từng node \(n\), gate nhận hai representation của chính node đó:

\[
g_n=\sigma(W_g[h_n^f\Vert h_n^m]+b_g)
\in(0,1)^q.
\]

\[
h_n^{UI}=g_n\odot h_n^f+(1-g_n)\odot h_n^m.
\]

Gate là vector theo từng dimension. Code dùng chung một gate network cho user và item, nhưng áp dụng **độc lập theo từng hàng**: user chỉ fusion với cùng user, item chỉ fusion với cùng item. Gate không concatenate user với item.

Output sau gated-sum vẫn có \(q\) chiều, không tăng thành \(2q\).

## 7. Parallel multimodal I–I graph

Từ feature image/text ban đầu, model xây dựng hai normalized KNN item graph rồi trộn:

\[
A_{MM}=\alpha A_v+(1-\alpha)A_t,
\qquad \alpha=\texttt{mm\_image\_weight}.
\]

Graph này được xây dựng/load từ cache và giữ cố định khi train. Tuy nhiên, item input \(E_i\) vẫn có thể học:

\[
H_{MM}=A_{MM}^{L_{MM}}E_i.
\]

I–I nhận **multimodal item input ban đầu**, không nhận item output U–I đã fusion. Hai đường truyền chạy song song:

```text
User input full ─┐
Shared item input ── Full U–I ──── H_full ─┐
                                           ├─ gated-sum ── U_final
User input mask ─┐                         │               I_UI ────┐
Shared item input ── Masked U–I ── H_mask ─┘                        ├─ I_final
                                                                   │
Shared multimodal item input ── Frozen I–I graph ── H_MM ──────────┘

                H_full ↔ H_mask: symmetric CL trước fusion
```

Representation cuối:

\[
U_{final}=U_{UI},
\qquad
I_{final}=I_{UI}+H_{MM}.
\]

## 8. Symmetric CL giữa hai branch

Trong mỗi training batch, lấy tập unique user \(\mathcal B_u\) và unique positive item \(\mathcal B_i\). Negative item được sample cho BPR không tự động được thêm vào tập item CL.

Với hai view \(X,Y\) của cùng tập node \(\mathcal B\), đặt:

\[
s(x,y)=\frac{x^\top y}{\|x\|_2\|y\|_2},
\]

\[
\ell(X\rightarrow Y)
=-\frac{1}{|\mathcal B|}\sum_{a\in\mathcal B}
\log\frac{\exp(s(x_a,y_a)/\tau)}
{\sum_{b\in\mathcal B}\exp(s(x_a,y_b)/\tau)}.
\]

Symmetric InfoNCE là:

\[
\operatorname{InfoNCE}_{sym}(X,Y)
=\frac12[\ell(X\rightarrow Y)+\ell(Y\rightarrow X)].
\]

Loss giữa hai branch:

\[
L_{branch}=\frac12\left[
\operatorname{InfoNCE}_{sym}(U_f[\mathcal B_u],U_m[\mathcal B_u])
+\operatorname{InfoNCE}_{sym}(I_f[\mathcal B_i],I_m[\mathcal B_i])
\right].
\]

Cùng một user/item giữa hai branch là positive pair; các node khác trong tập CL là negative. Không đối chiếu user với item.

CL được tính trên U–I output trước fusion, không bao gồm I–I residual. Không có `stopgrad`: gradient đi vào cả hai user branch, các modality input/projection dùng chung và learned mask qua masked propagation. CL không trực tiếp đi qua fusion gate hay I–I output.

## 9. Recommendation loss và mask regularization

Điểm recommendation:

\[
\widehat y_{ui}=u_{final,u}^\top i_{final,i}.
\]

Với training triplet \((u,i^+,i^-)\), BPR khuyến khích positive item có điểm cao hơn negative item:

\[
L_{BPR}=-\frac1{|\mathcal T|}\sum_{(u,i^+,i^-)\in\mathcal T}
\log\sigma(\widehat y_{ui^+}-\widehat y_{ui^-}).
\]

Mask regularization trên toàn bộ \(|\mathcal E|\) cạnh train:

\[
L_{budget}=\left(\frac1{|\mathcal E|}\sum_{e\in\mathcal E}m_e-\rho\right)^2,
\]

\[
L_{binary}=\frac1{|\mathcal E|}\sum_{e\in\mathcal E}m_e(1-m_e),
\qquad
L_{mask}=L_{budget}+\beta L_{binary}.
\]

Budget loss khuyến khích trung bình mask gần \(\rho\); binary loss khuyến khích mask gần 0 hoặc 1. Cả hai là soft penalty, không đảm bảo sparsity hay chất lượng lựa chọn cạnh.

Loss tổng:

\[
\boxed{L=L_{BPR}+\lambda_{CL}L_{branch}+\lambda_{mask}L_{mask}}
\]

Trong code, \(\lambda_{CL}=\texttt{cl\_weight}\), \(\lambda_{mask}=\texttt{mask\_weight}\), \(\beta=\texttt{mask\_binary\_weight}\).

BPR học ranking và fusion; CL căn chỉnh hai view; mask regularization kiểm soát mức trọng số cạnh. Mask nhận tín hiệu lựa chọn cạnh từ BPR và CL, không có nhãn trực tiếp cho cạnh tốt/xấu.

## 10. Inference và giới hạn của ý tưởng

Inference giữ full train graph và learned soft-mask graph, dùng gate đã học, cộng I–I residual rồi xếp hạng theo dot product. Không tính CL hay loss khi inference; không đưa validation/test interaction vào graph.

Nếu thay `soft` bằng `hard`, masked branch giữ global top-k cạnh: train dùng Gumbel top-k và straight-through gradient trên cạnh được chọn; eval dùng deterministic top-k theo learned logits. Khi đó số cạnh giữ lại được kiểm soát trực tiếp, nhưng không đảm bảo mỗi user đều còn hàng xóm.

Mục tiêu của mô hình là kết hợp một view đầy đủ với một view học điều chỉnh interaction. CL giúp chúng tương thích để fusion, nhưng không đảm bảo masked branch học graph tốt hơn random mask hoặc mô hình luôn vượt baseline. Các giả thuyết đó cần được xác nhận bằng ablation và nhiều seed.
