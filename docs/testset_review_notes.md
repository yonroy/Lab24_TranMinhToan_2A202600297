# Test Set Review Notes — Phase A.1

Ngày review: ___________  
Reviewer: ___________  
Tổng số câu trong testset: 50  
Số câu đã review thủ công: 10 (rows được chọn: _____)

---

## Phân phối (Distribution Check)

```
evolution_type
simple         25  (50%)
reasoning      13  (26%)
multi_context  12  (24%)
```

_Ghi lại output thực tế từ `df['evolution_type'].value_counts()` vào đây._

---

## Review Log (10 câu đại diện)

### Câu 1

| Trường | Nội dung |
|--------|----------|
| **Question** | _(dán question vào đây)_ |
| **Ground truth** | _(dán ground_truth vào đây)_ |
| **Evolution type** | simple / reasoning / multi_context |
| **Đánh giá** | ☐ Hợp lệ &nbsp; ☐ Cần chỉnh sửa &nbsp; ☐ Loại bỏ |
| **Vấn đề** | _(mô tả nếu có)_ |
| **Chỉnh sửa** | _(ghi nội dung sau khi sửa, nếu có)_ |

---

### Câu 2

| Trường | Nội dung |
|--------|----------|
| **Question** | |
| **Ground truth** | |
| **Evolution type** | |
| **Đánh giá** | ☐ Hợp lệ &nbsp; ☐ Cần chỉnh sửa &nbsp; ☐ Loại bỏ |
| **Vấn đề** | |
| **Chỉnh sửa** | |

---

### Câu 3

| Trường | Nội dung |
|--------|----------|
| **Question** | |
| **Ground truth** | |
| **Evolution type** | |
| **Đánh giá** | ☐ Hợp lệ &nbsp; ☐ Cần chỉnh sửa &nbsp; ☐ Loại bỏ |
| **Vấn đề** | |
| **Chỉnh sửa** | |

---

### Câu 4

| Trường | Nội dung |
|--------|----------|
| **Question** | |
| **Ground truth** | |
| **Evolution type** | |
| **Đánh giá** | ☐ Hợp lệ &nbsp; ☐ Cần chỉnh sửa &nbsp; ☐ Loại bỏ |
| **Vấn đề** | |
| **Chỉnh sửa** | |

---

### Câu 5

| Trường | Nội dung |
|--------|----------|
| **Question** | |
| **Ground truth** | |
| **Evolution type** | |
| **Đánh giá** | ☐ Hợp lệ &nbsp; ☐ Cần chỉnh sửa &nbsp; ☐ Loại bỏ |
| **Vấn đề** | |
| **Chỉnh sửa** | |

---

### Câu 6

| Trường | Nội dung |
|--------|----------|
| **Question** | |
| **Ground truth** | |
| **Evolution type** | |
| **Đánh giá** | ☐ Hợp lệ &nbsp; ☐ Cần chỉnh sửa &nbsp; ☐ Loại bỏ |
| **Vấn đề** | |
| **Chỉnh sửa** | |

---

### Câu 7

| Trường | Nội dung |
|--------|----------|
| **Question** | |
| **Ground truth** | |
| **Evolution type** | |
| **Đánh giá** | ☐ Hợp lệ &nbsp; ☐ Cần chỉnh sửa &nbsp; ☐ Loại bỏ |
| **Vấn đề** | |
| **Chỉnh sửa** | |

---

### Câu 8

| Trường | Nội dung |
|--------|----------|
| **Question** | |
| **Ground truth** | |
| **Evolution type** | |
| **Đánh giá** | ☐ Hợp lệ &nbsp; ☐ Cần chỉnh sửa &nbsp; ☐ Loại bỏ |
| **Vấn đề** | |
| **Chỉnh sửa** | |

---

### Câu 9

| Trường | Nội dung |
|--------|----------|
| **Question** | |
| **Ground truth** | |
| **Evolution type** | |
| **Đánh giá** | ☐ Hợp lệ &nbsp; ☐ Cần chỉnh sửa &nbsp; ☐ Loại bỏ |
| **Vấn đề** | |
| **Chỉnh sửa** | |

---

### Câu 10 ✎ (câu được chỉnh sửa — bắt buộc ≥1)

| Trường | Nội dung |
|--------|----------|
| **Question gốc** | _(dán question gốc do LLM gen)_ |
| **Ground truth gốc** | _(dán ground truth gốc)_ |
| **Evolution type** | |
| **Vấn đề phát hiện** | _(ví dụ: câu hỏi không liên quan domain, ground truth sai thực tế, ambiguous)_ |
| **Question sau chỉnh sửa** | _(nội dung đã sửa)_ |
| **Ground truth sau chỉnh sửa** | _(nội dung đã sửa)_ |
| **Lý do chỉnh sửa** | _(giải thích tại sao cần sửa)_ |

---

## Tổng kết

| Hạng mục | Số lượng |
|----------|----------|
| Câu hợp lệ | X / 10 |
| Câu cần chỉnh sửa | X / 10 |
| Câu loại bỏ | X / 10 |

**Nhận xét chung về chất lượng test set:**

_(Ví dụ: "Các câu simple nhìn chung tốt. Câu reasoning đôi khi yêu cầu inference phức tạp hơn corpus cho phép.
Multi-context questions có xu hướng gen ra câu chung chung, không cụ thể domain.")_

**Hành động tiếp theo:**

- [ ] Regenerate X câu bị loại bỏ
- [ ] Xác nhận distribution lại sau khi sửa
- [ ] Proceed to Task A.2
