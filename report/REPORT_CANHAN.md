# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Lê Minh Sang
**Nhóm:** 3Kings
**Ngày:** 2026-09-20

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector có hướng gần nhau trong không gian embedding, nên hai đoạn text có quan hệ ngữ nghĩa gần nhau. Điểm cao không đòi hỏi hai câu phải dùng đúng cùng từ.

**Ví dụ có độ tương tự CAO:**

- Câu A: Người mua cần chờ 48 giờ sau Help request trước khi mở case.
- Câu B: Sau khi liên hệ seller được hai ngày, buyer mới có thể yêu cầu Etsy can thiệp.
- Tại sao tương đồng: Cả hai diễn đạt cùng điều kiện thời gian để escalation, dù dùng từ vựng khác nhau.

**Ví dụ có độ tương tự THẤP:**

- Câu A: Seller chọn Refund trong Shop Manager để hoàn tiền đơn hàng.
- Câu B: Estimated delivery date được tạo từ processing time và carrier transit time.
- Tại sao khác: Một câu nói về thao tác hoàn tiền, câu kia nói về cách tính thời gian giao hàng; mục tiêu và thông tin cần truy xuất khác nhau.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine đo góc giữa hai vector nên tập trung vào hướng ngữ nghĩa và ít bị ảnh hưởng bởi độ lớn vector. Điều này phù hợp retrieval text, nơi độ dài đoạn văn không nên tự làm hai nội dung cùng nghĩa trở nên xa nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Phép tính: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11...)`.
> Đáp án: **23 chunks**; đã kiểm lại bằng `FixedSizeChunker(chunk_size=500, overlap=50)` trên chuỗi 10.000 ký tự.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> `ceil((10000 - 100) / (500 - 100)) = 25`, nên số chunk tăng từ 23 lên **25**. Overlap lớn hơn giữ ngữ cảnh ở ranh giới chunk tốt hơn, đổi lại tạo thêm embedding và chi phí lưu trữ/truy xuất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Hàm dùng `re.split(r"(?<=[.!?])\s+", text.strip())`: lookbehind tách ngay sau dấu kết câu nên không làm mất `.`, `!`, `?`. Các câu được `strip`, rồi ghép từng nhóm tối đa `max_sentences_per_chunk`; text rỗng hoặc chỉ có khoảng trắng trả `[]`. Đây là heuristic của Lab, nên chữ viết tắt như `TS.` hoặc số thập phân vẫn là giới hạn đã biết.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> `_split` lần lượt thử `\n\n`, `\n`, `. `, khoảng trắng rồi cắt cứng; chỉ mảnh vượt `chunk_size` mới hạ xuống separator ít ưu tiên hơn. Base case là mảnh đủ nhỏ, hết separator, hoặc separator rỗng; hai trường hợp sau cắt theo `chunk_size`. Separator được giữ trong lúc đệ quy và `_merge_fragments` gom các mảnh kề nhau để tránh chunk vụn.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Store dùng in-memory để hành vi ổn định. `_make_record` tạo embedding, copy metadata và bảo đảm có `metadata["doc_id"]`; id chunk như `buyer-open-case#2` tự suy ra doc_id cha. `add_documents` giữ quy ước 1 `Document` = 1 record; `search` dùng cosine similarity, sắp xếp giảm dần theo score và không trả vector embedding dài.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` lọc toàn bộ cặp key/value của metadata **trước** khi ranking, nên chunk sai audience không chiếm chỗ top-k. Khi không có filter, nó dùng cùng đường search chuẩn. `delete_document` loại mọi record có `metadata["doc_id"]` trùng id yêu cầu và so sánh kích thước trước/sau để trả `True` hoặc `False`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Agent retrieve top-k rồi dựng context theo dạng `[1] Source: ...`; source ưu tiên `source_url`, sau đó `source` hoặc `doc_id` để truy vết. Prompt yêu cầu chỉ dùng context, trích block nguồn và nói rõ khi không có đáp án. Nếu store rỗng hoặc `top_k=0`, hàm trả thông báo không tìm thấy mà không gọi `llm_fn`; implementation này được unit test, còn benchmark mục 5 chỉ đánh giá retrieval/extractive evidence.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ .venv/bin/python -m pytest tests/ -v
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.1.1
collecting ... collected 42 items
============================== 42 passed in 0.11s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Buyer must wait 48 hours after a Help request before opening a case. | After messaging the seller for two days, a buyer may ask Etsy to resolve an order issue. | cao | 0.7418 | Có |
| 2 | A seller opens Shop Manager, chooses Orders, and selects Refund. | To issue a full or partial refund, the seller selects an order and enters the refund amount. | cao | 0.8210 | Có |
| 3 | Estimated delivery combines processing time and carrier transit time. | A seller can refund an order through Etsy Payments within 180 days. | thấp | 0.5623 | Có |
| 4 | Purchase Protection can refund a qualifying order that never arrives. | Eligible buyers receive a full refund when their item is lost, damaged, or significantly different from the listing. | cao | 0.7266 | Có |
| 5 | A return policy can be applied to multiple physical listings. | A buyer should provide photos when an item arrived damaged in a case. | thấp | 0.6653 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Các điểm được đo trước bằng FastEmbed 0.8.0, model `BAAI/bge-small-en-v1.5` (384 chiều), rồi gọi `compute_similarity()` trực tiếp, không dùng MockEmbedder. Cặp 5 gây bất ngờ nhất: khác hành động chính nhưng vẫn đạt `0.6653`, cho thấy embedding giữ bối cảnh chung buyer/item/physical listing. Tuy vậy, cặp 2 cùng thao tác refund vẫn cao nhất (`0.8210`), nên vector biểu diễn mức gần nghĩa theo ngữ cảnh chứ không chỉ kiểm tra intent trùng khớp.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân trong gói `src`, bằng `FixedSizeChunker(chunk_size=600, overlap=100)`, FastEmbed 0.8.0 `BAAI/bge-small-en-v1.5`, `top_k=3`. Chunks giữ frontmatter và `doc_id`; output tái lập được đã lưu ở `ket_qua_benchmark.txt`.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What conditions must be met before a buyer can open a case on Etsy? | `buyer-open-case#0`: Quick answer chứa estimated delivery date và `48 hours`. | 0.833737 | Có — gold doc và marker top-1. | Extractive: estimated delivery date phải qua và buyer đã liên hệ seller hơn 48 giờ mới mở case. |
| 2 | How does a seller work with a buyer to resolve an open Etsy case through Shop Manager? | `seller-resolve-case#4`: Cases in Shop Manager, case ID, Add Your Comment. | 0.885760 | Có — gold doc và marker top-1. | Extractive: vào Shop Manager → Help → Cases, chọn case và trao đổi qua Add Your Comment. |
| 3 | How much refund does Etsy Purchase Protection provide for a qualifying order? | `buyer-purchase-protection#0`: full refund for qualifying issues. | 0.896129 | Có — gold doc và `full refund` top-1. | Extractive: buyer nhận full refund khi đơn đủ điều kiện. |
| 4 | Which components are used to calculate an Etsy estimated delivery date? | `buyer-estimated-delivery#3`; formula ở `buyer-estimated-delivery#1` (top-3). | 0.837842 | Có — gold doc top-1; marker top-3. | Extractive: processing time + carrier transit time = estimated delivery date. |
| 5 | How can a seller issue a full or partial refund, and what is the Etsy Payments time limit? | `seller-issue-refund#0`; mốc thời gian ở `#4` (top-2). | 0.882164 | Có — gold doc top-1; `180 days` top-2. | Extractive: refund qua Shop Manager sau khi payment processed và trước 180 ngày. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> A/B của Hoàng cho thấy một query có cùng từ khoá có thể có hai đáp án đúng khác nhau theo audience: buyer là `full refund`, seller là `$250`. Vì vậy chỉ kiểm `doc_id` là chưa đủ; cần kiểm marker/answer span và filter `audience` trước ranking. FixedSize + overlap giữ evidence trong top-3 ở cả 5 query, nhưng Q4/Q5 cũng cho thấy cần đọc toàn bộ top-3 thay vì chỉ nhìn top-1.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
