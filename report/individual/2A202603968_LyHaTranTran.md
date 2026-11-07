# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603968
- **Họ và tên:** Lý Hà Trần Trân
- **SEED:** 7144

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Bộ kiểm tra tự động: **15/15 đạt** (gồm 5/5 phép tính tay: `baseline_accuracy = 0.87`, `sigmoid_0 = 0.5`, `f1_example = 0.73`, `cosine_example = 0.6`, `llm_cost_per_day = 10.5`).

## 2. Kết quả chính

Dữ liệu sau khi bỏ trùng: 5.169 tin (653 spam, 4.516 ham); train 3.619 · val 775 · test 775 (116 spam) · SUB 40 tin (20 spam + 20 ham).

| Cách tiếp cận | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | :---: | :---: | :---: | :---: |
| 1. BoW + LogReg (tự code) | 0.834 | 0.621 | 0.08 | 0 |
| 2. BoW + MLP | 0.926 | 0.889 | 10.34 | 0 |
| 3a. MiniLM + kNN | 0.919 | 0.947 | 19.33 | 0 |
| 3b. MiniLM + LogReg | 0.946 | 0.889 | 7.43 | 0 |
| 3c. MiniLM + MLP head | **0.960** | 0.919 | 10.94 | 0 |
| 4a. LLM nhỏ zero-shot (`gpt-oss-20b`, Groq) | — | 0.927 | 2 435.56 | 0* |
| 4b. LLM lớn zero-shot (`gpt-oss-120b`, Groq) | — | **0.974** | 2 412.90 | 0* |
| 4c. LLM nhỏ few-shot | — | 0.923 | 2 461.60 | 0* |

\* Em chạy Phần 4 trên Groq (gói miễn phí); model `gpt-oss` không có trong bảng giá của notebook nên cột USD ghi 0. Nếu tính theo giá tham khảo của `gpt-6-luna` (100 token vào + 1 token ra mỗi tin, $0.10 / $0.50 mỗi 1M token) thì khoảng **$0.0105 / 1k tin**; `gpt-6.1-terra` đắt gấp 10 lần.

Ghi chú khi đọc bảng:
- LLM chỉ chạy trên SUB (40 tin), nên so chất lượng công bằng phải dùng cột *F1 (SUB 40)*. Với 40 tin, sai thêm 1 tin làm 1 − F1 đổi khoảng 2,5 điểm %; chênh dưới ~0,05 F1 coi như hoà.
- Đạt SLA p95 < 50 ms: cả 5 cách không phải LLM. Cả 3 cách LLM đều p95 ≈ 2,4 s và chỉ khoảng 1,2 tin/s khi gọi tuần tự.
- Trên đường Pareto (1 − F1 theo SUB và p95): BoW + LogReg, MiniLM + LogReg, MiniLM + MLP head, MiniLM + kNN và LLM lớn.

## 3. Dự đoán và thực tế

1. **`lr = 0.001` và `lr = 50`.** Em dự đoán `lr = 0.001` hội tụ chậm — khớp. Với `lr = 50` thì không khớp: em đã tưởng bước nhảy quá lớn chỉ làm loss tăng vọt. Thực tế do bài toán lồi, bước nhảy lớn đẩy trọng số `w` lên rất lớn, sigmoid bão hoà và loss trên tập train bị ép về sát 0, tức là mô hình học vẹt chứ không dao động mãi.
2. **`dl_relu`: đoán 0.85, thực tế 0.922.** Em tưởng bỏ ReLU thì mạng quay về mức của LogReg ở Phần 1 (0.834). Kết quả gần như không đổi so với MLP có ReLU (0.926). Điều này cho thấy trên BoW, tính phi tuyến của ReLU đóng góp rất ít; phần cải thiện so với Phần 1 nhiều khả năng đến từ cách huấn luyện (Adam, mini-batch, 30 epoch, nhiều tham số hơn) chứ không phải từ lớp ẩn phi tuyến.
3. **`emb_tricky`: đoán Encoder + LogReg đúng nhiều hơn, thực tế ngược lại** — BoW đúng 5/6, Encoder + LogReg đúng 4/6. BoW bắt được tin "Your mobile number has been chosen for a complimentary holiday…" nhờ các từ như *your*, *mobile*, *number*, *chosen*, còn Encoder + LogReg xếp nó về phía ham. Cả hai đều bỏ sót tin "Last chance! Your account rewards expire tonight…". Em đánh giá quá cao embedding ở những tin spam văn phong lịch sự.

## 4. Phân tích 50 case khó

Em chạy bộ 50 case gốc cộng thêm 7 case tự viết (mỗi nhóm 1 case), tổng 57 case.

| Mô hình | F1 test thường (SUB) | F1 case khó | Tụt | Accuracy case khó |
| --- | :---: | :---: | :---: | :---: |
| BoW + LogReg | 0.621 | 0.341 | −0.280 | 0.526 |
| BoW + MLP | 0.889 | 0.465 | **−0.424** | 0.596 |
| MiniLM + kNN | 0.947 | 0.727 | −0.220 | 0.684 |
| MiniLM + LogReg | 0.889 | 0.586 | −0.303 | 0.579 |
| MiniLM + MLP head | 0.919 | 0.633 | −0.286 | 0.614 |
| LLM nhỏ zero-shot | 0.927 | 0.870 | −0.057 | 0.842 |
| LLM nhỏ + prompt phòng thủ | 0.927* | 0.901 | −0.026 | 0.877 |

\* Mốc test thường của hàng này lấy từ LLM nhỏ zero-shot.

- **Mô hình tụt nhiều nhất: BoW + MLP** (−0.424), rồi MiniLM + LogReg (−0.303) và MiniLM + MLP head (−0.286). LLM tụt ít nhất. Điểm test thường quá lạc quan vì tập test cắt ngẫu nhiên từ cùng nguồn với train và chỉ bỏ trùng tuyệt đối: spam mẫu chỉ khác số điện thoại hay mã thưởng vẫn nằm ở cả hai tập.
- **BoW khó nhất ở spam không có từ khoá quen:** spam giọng tự nhiên chỉ đúng 9% (LogReg) và 0% (MLP), né từ khoá 14%, thao túng/lừa đảo 22% và 44%. BoW gần như luôn đoán ham ở bộ này, nên nó "đúng 100%" ở các nhóm ham chỉ vì mặc định đoán ham.
- **Embedding** bắt tốt spam né từ khoá (LogReg và MLP head đúng 100%, kNN 86%) vì tokenizer subword của MiniLM vẫn giữ được phần nghĩa của chữ viết biến dạng. Nhưng thao túng/lừa đảo chỉ đúng 11% (LogReg) và 33% (MLP head), spam giọng tự nhiên 27%. kNN yếu ở nhóm "nói về spam" (20%).
- **LLM** đúng 100% ở thao túng/lừa đảo, né từ khoá và prompt injection; lỗi tập trung ở ham chứa từ khoá spam (64%) và "nói về spam" (80%).
- **Case 1 (ham bị đoán spam):** #2 *"FREE pizza in the office kitchen right now, come quick before it's gone"* — chứa "FREE", "now", giục giã nhưng là tin đồng nghiệp. LLM nhỏ cả hai bản prompt đều đoán spam.
- **Case 2 (spam bị lọt):** #13 *"Hope you're well! Quick reminder that your photos from the event are ready, view & order prints here: snap-prints.co/r/88"* — là case duy nhất làm **tất cả** model sai: giọng nhắc nhở thân thiện, không có từ khoá, link bán hàng không ai yêu cầu.
- **Prompt injection:** LLM đúng 100% ở nhóm này với cả prompt thường và prompt phòng thủ (thẻ `<sms>`), nên số case chưa đủ để kết luận là an toàn. Prompt phòng thủ nâng spam giọng tự nhiên từ 73% lên 91% và tổng từ 84% lên 88% (khoảng 2 case, gần mức nhiễu), nhưng đổi cả định nghĩa spam/ham nên không tách được tác dụng của riêng phần chống injection. Vẫn cần lớp bảo vệ ngoài prompt (luật bắt mẫu "ignore previous instructions", không để LLM là lớp kiểm tra duy nhất).
- **Bộ 50 case do người viết tay**, toàn tiếng Anh và mỗi case ≈ 2 điểm %, nên chỉ dùng để tìm điểm mù và so sánh kiểu lỗi, không dùng để báo cáo độ chính xác chung.

## 5. Thử thách cá nhân

**Đề #5 (SEED 7144 % 6 = 4):** Embedding — viết 4 tin của riêng mình (2 spam né từ khoá, 2 ham chứa "free"/"win"), so dự đoán BoW + LogReg với Embedding + LogReg và giải thích từng tin.

| # | Loại tin | Tin nhắn | Thật | BoW + LogReg | Embedding + LogReg |
| :-: | --- | --- | :-: | :-: | :-: |
| 1 | Spam né từ khoá | *Hi friend, I saw your resume online and have an open remote role at 40 bucks an hour. Reach my assistant on tg @joblead* | Spam | Ham ✗ | Ham ✗ |
| 2 | Spam né từ khoá | *Your parcel delivery could not be scheduled due to an incomplete street number. Please update your address at bit.ly/pkg-track-9* | Spam | Ham ✗ | Ham ✗ |
| 3 | Ham chứa "free" | *Are you free this Saturday afternoon to review the lecture notes for our exam?* | Ham | Ham ✓ | Ham ✓ |
| 4 | Ham chứa "win" | *Did our university team win the basketball match yesterday? Tell me the final score!* | Ham | Ham ✓ | Ham ✓ |

Cả hai mô hình đều đúng 2/4: đúng cả hai tin ham, sai cả hai tin spam.

**Giải thích:**
- **Tin 1, 2 (cả hai đều bỏ sót):** BoW không thấy từ khoá spam nào có trọng số dương lớn (*free, win, prize, claim, txt…*); chỉ có các từ trung tính như *resume, remote, role, parcel, delivery*, nên tổng `w·x + b` vẫn âm và tin bị coi là ham. Embedding + LogReg cũng sai vì đầu LogReg chỉ học từ dữ liệu SMS spam cũ (khuyến mãi, tin nhắn tính phí). Tin tuyển dụng giả và tin "giao hàng thất bại + link rút gọn" mang văn phong lịch sự, nên vector nằm gần vùng ham hơn. Điều này khớp với kết quả ở 50 case khó: embedding yếu ở spam giọng tự nhiên (27%) và thao túng/lừa đảo (11%).
- **Tin 3, 4 (cả hai đều đúng):** BoW không bị từ "free" hay "win" đánh lừa, vì các từ ham mạnh trong câu (*are, you, this, our, did, me*…) kéo tổng điểm xuống, một từ khoá đơn lẻ chưa đủ vượt ngưỡng. Embedding hiểu "free" là rảnh và "win" ở ngữ cảnh trận đấu giữa bạn bè. Điều này khớp với thí nghiệm `emb_tricky`, nơi BoW cũng đúng cả các tin ham chứa từ khoá.
- **Kết luận:** với 4 tin tự viết, giả thuyết "embedding bắt tốt hơn BoW" **không được chứng minh**. Embedding chỉ hơn rõ ở né từ khoá bằng *biến dạng chữ* (`c0ngr4tulat10ns`, `F R E E`), không phải ở spam né từ khoá bằng *ngữ nghĩa*. Mẫu 4 tin quá nhỏ để khái quát.

## 6. Phương án cho sản phẩm thật

Yêu cầu p95 < 50 ms/tin, 10 triệu SMS/ngày (≈ 116 tin/giây trung bình).

- **Chọn:** MiniLM embedding + MLP head (3c: F1 0.960, precision 0.991, recall 0.931, p95 10.94 ms, ≈ 2 150 tin/s cả lô) làm tầng chính. MiniLM + LogReg (3b: F1 0.946, p95 7.43 ms) là phương án nhẹ hơn nếu cần độ trễ thấp hơn. kNN (p95 19,3 ms, 140 tin/s cả lô) chậm nhất nhóm và phải giữ toàn bộ vector train trong RAM nên không hợp với dữ liệu hàng triệu tin.
- **Vì sao không dùng LLM toàn bộ:** p95 ≈ 2,4 s (gấp ~50 lần SLA), 1,2 tin/s khi gọi tuần tự, nên để lọc 116 tin/giây cần hàng trăm request chạy song song. Chi phí theo giá tham khảo của notebook (100 token vào + 1 token ra): `gpt-6-luna` ≈ $105/ngày ≈ **$3,150/tháng**; `gpt-6.1-terra` ≈ **$31,500/tháng**. Đây là mức tối thiểu vì prompt thật dài hơn 100 token.
- **Thiết kế lai (router):**
  1. Tầng 0 — chuẩn hoá và luật (`0→o`, `€→e`, gộp chữ bị tách; bắt mẫu injection; kiểm tra URL/đầu số lạ).
  2. Tầng 1 — MiniLM + MLP head; tin có xác suất xa ngưỡng (p < 0.3 hoặc p > 0.7) thì quyết luôn.
  3. Tầng 2 — LLM với prompt phòng thủ chỉ nhận ~5% tin lưỡng lự hoặc bị tầng 0 đánh dấu, chạy bất đồng bộ ngoài đường 50 ms. Chi phí còn khoảng $5/ngày ≈ $160/tháng với model nhỏ.
- **Đánh đổi và rủi ro:** chấp nhận F1 thấp hơn LLM ở case khó (model rẻ chỉ đúng 58–68% trên 57 case khó, LLM 84–88%), đổi lại độ trễ và chi phí thấp; thêm độ phức tạp vận hành (ngưỡng router, giám sát drift). Router chỉ hiệu quả nếu các tin model rẻ sai là tin nó *không chắc chắn*. Với thao túng/lừa đảo, Encoder + LogReg chỉ đúng 11%, nên cần đo xem các lần sai đó có rơi vào vùng lưỡng lự hay không, và bổ sung tín hiệu định tuyến khác (luật tầng 0, sự bất đồng giữa hai model rẻ). Tỉ lệ 5% là giả định, cần đo trên dữ liệu thật.

## 7. AI log và điều đã học

| Phần | Em hỏi | AI trả lời | Em sửa/kiểm |
| --- | --- | --- | --- |
| 4.1 | Cách viết `call_openai` gọi qua Groq với `provider_kwargs`, xử lý lỗi 400 (`reasoning_effort`) và 404 (`model_not_found`) | Cấu hình OpenAI client trỏ về endpoint Groq; gợi ý map `gpt-6-luna` sang model có sẵn (`openai/gpt-oss-20b`); dùng `reasoning_effort="low"` thay `"none"` | Tự liệt kê model khả dụng bằng `client.models.list()`, xác nhận tài khoản có `openai/gpt-oss-20b`, sửa tham số và chạy qua assert test của ô 4.1 |
| 4.2 | Hoàn thiện `build_classification_prompt` cho zero-shot/few-shot | Template gồm mô tả nhiệm vụ, yêu cầu trả đúng một từ `spam`/`ham`, vòng lặp chèn ví dụ few-shot và phần đuôi `Message: """..."""\nLabel:` | Đối chiếu cú pháp dấu ba nháy và xuống dòng với docstring, chạy thử với một tin mẫu |
| 4.4 | Viết `llm_classify_batch` kết hợp `retry_with_backoff`, `parse_label` và ước tính chi phí | Khung hàm duyệt từng tin, gọi `call_llm` qua lambda retry, ép nhãn không hợp lệ về 0 và đếm `n_invalid`, tính token bằng `số_từ / 0.75` | Đảm bảo `preds` là `np.ndarray` để dùng được với `metrics`; chạy thử trên tập SUB 40 tin |


**Điều em hiểu ra:**
1. **Dự đoán sai nhiều nhất là hình dạng đường loss khi đổi learning rate** (và cả việc ReLU giúp được bao nhiêu). Learning rate ảnh hưởng rất mạnh đến quá trình huấn luyện, nên phải chọn theo từng bài; không nên đoán bằng trực giác.
2. Điểm cao trên tập test thường chưa chứng minh mô hình "hiểu" spam: BoW + MLP có F1 0.926 nhưng tụt còn 0.465 trên case khó, vì chỉ nhận lại từ khoá đã thấy khi train.
3. Chọn mô hình cho sản phẩm là bài toán đánh đổi giữa F1, độ trễ và chi phí chứ không chỉ chọn F1 cao nhất. Một mô hình nhanh rẻ làm tầng chính, kèm tầng LLM cho tin khó, hợp lý hơn việc chọn một cách duy nhất.