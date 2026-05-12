# LLM Judge Bias Report — Phase B.4

Judge model: `gpt-4o-mini`  
Dataset: `pairwise_results.csv` — X pairs  
Ngày phân tích: ___________

---

## Tóm tắt

| Bias | Mức độ | Có đáng lo? |
|------|--------|-------------|
| Position bias | X% (A wins when listed first) | ☐ Có (>55%) &nbsp; ☐ Không |
| Length bias | X/Y cases B wins when longer | ☐ Có &nbsp; ☐ Không |
| _(bias khác nếu có)_ | | |

Cohen's kappa (human vs judge): **X.XX** — _(interpretation: slight / fair / moderate / substantial / almost perfect)_

---

## Bias 1 — Position Bias

**Định nghĩa:** Judge có xu hướng chọn answer được đưa vào trước (vị trí A) dù chất lượng thực tế tương đương.

**Phương pháp đo:**
```python
run1_a_wins = (df['run1_winner'] == 'A').sum()
total = len(df)
position_bias_rate = run1_a_wins / total
# Expected ~50% nếu không có bias. >55% = có bias đáng kể.
```

**Kết quả:**

| Chỉ số | Giá trị |
|--------|---------|
| Tổng pairs | X |
| A wins (run1, A listed first) | X |
| B wins (run1, B listed first) | X |
| Tie (run1) | X |
| **A win rate khi listed first** | **X.X%** |
| Expected nếu không bias | ~50% |
| Chênh lệch | +X.X% |

**Biểu đồ:**

```
Position bias — A win rate khi listed first

60% |████████████████████████████ A wins
50% |-------------------------- (expected)
    |
    Run 1 (A first)    Run 2 (B first, flipped)
    A wins: X%         A wins: X% (after flip)
```

> _(Thay thế bằng matplotlib chart thực tế nếu có — lưu vào phase-b/bias_position_chart.png)_

**Kết luận:**

- Nếu X% > 55%: Judge có position bias. Swap-and-average đã mitigate: tỉ lệ thay đổi như thế nào sau khi aggregate run1 + run2?
- Nếu X% ≈ 50%: Không có position bias đáng kể.

**Sau khi apply swap-and-average:**

| Chỉ số | Giá trị |
|--------|---------|
| Final A wins | X |
| Final B wins | X |
| Final ties | X |
| Tie rate tăng so với run1 | +X% (expected: swap tạo ra nhiều ties hơn) |

---

## Bias 2 — Length Bias

**Định nghĩa:** Judge có xu hướng chọn answer dài hơn dù không nhất thiết tốt hơn về chất lượng.

**Phương pháp đo:**
```python
df['len_a'] = df['answer_a'].str.len()
df['len_b'] = df['answer_b'].str.len()
df['len_diff'] = df['len_b'] - df['len_a']  # dương = B dài hơn

b_wins_when_longer = ((df['winner_after_swap'] == 'B') & (df['len_diff'] > 0)).sum()
b_total_longer = (df['len_diff'] > 0).sum()
b_wins_when_shorter = ((df['winner_after_swap'] == 'B') & (df['len_diff'] < 0)).sum()
b_total_shorter = (df['len_diff'] < 0).sum()
```

**Kết quả:**

| Điều kiện | B wins | Total | Win rate |
|-----------|--------|-------|----------|
| B dài hơn A | X | X | X.X% |
| B ngắn hơn A | X | X | X.X% |
| Chênh lệch | | | +X.X% |

**Phân tích theo mức độ chênh lệch độ dài:**

| Khoảng len_diff | B wins | Total | Win rate |
|-----------------|--------|-------|----------|
| B dài hơn < 50 chars | X | X | X% |
| B dài hơn 50–200 chars | X | X | X% |
| B dài hơn > 200 chars | X | X | X% |

**Biểu đồ:**

```
Length bias — B win rate theo độ dài tương đối

     B win rate
80%  |                              ████
60%  |              ████    ████
40%  |   ████
20%  |
     +----------------------------------
     B ngắn hơn  Tương đương  B dài hơn
```

> _(Thay thế bằng matplotlib chart thực tế — lưu vào phase-b/bias_length_chart.png)_

**Kết luận:**

- Nếu B win rate tăng rõ khi B dài hơn (>60%): Judge có length bias đáng kể.
- Nếu tăng nhẹ (5–10%): Bias tồn tại nhưng không nghiêm trọng.

---

## Bias 3 — Verbosity / Style Bias (nếu phát hiện)

**Định nghĩa:** Judge ưu tiên answers dùng bullet points, headers, hoặc markdown formatting hơn prose thuần.

**Phương pháp đo:**
```python
df['a_has_bullets'] = df['answer_a'].str.contains(r'^\s*[-*]', regex=True)
df['b_has_bullets'] = df['answer_b'].str.contains(r'^\s*[-*]', regex=True)

# B wins khi B có bullets nhưng A không
b_wins_structured = ((df['winner_after_swap'] == 'B') 
                     & df['b_has_bullets'] 
                     & ~df['a_has_bullets']).sum()
```

**Kết quả:** _(Điền nếu thực hiện phân tích này)_

---

## Human Calibration — Cohen's Kappa

**Sample:** 10 pairs từ `pairwise_results.csv` (random_state=42)

| Pair | Human label | Judge label | Agreement |
|------|-------------|-------------|-----------|
| 1 | A | A | ✓ |
| 2 | B | B | ✓ |
| 3 | tie | A | ✗ |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

**Cohen's kappa: X.XX**

| Kappa | Interpretation |
|-------|----------------|
| < 0 | Worse than chance |
| 0.0–0.2 | Slight agreement |
| 0.2–0.4 | Fair agreement |
| 0.4–0.6 | Moderate agreement |
| **0.6–0.8** | **Substantial — production ready ✓** |
| 0.8–1.0 | Almost perfect |

**Kết quả của tôi:** kappa = X.XX → _(interpretation)_

**Root cause nếu kappa < 0.6:**

_(Ví dụ: "Judge có length bias rõ ràng — khi B dài hơn 2x, judge chọn B gần như tuyệt đối
dù human judges không nhận thấy sự khác biệt về chất lượng thực sự. Human thường đánh giá
cao brevity và relevance hơn, trong khi judge LLM bị thu hút bởi comprehensiveness.")_

---

## Mitigation Strategies

| Bias | Strategy đã áp dụng | Strategy đề xuất thêm |
|------|--------------------|-----------------------|
| Position bias | Swap-and-average (run mỗi cặp 2 lần, đổi thứ tự) | Chain-of-thought trước khi verdict |
| Length bias | — | Thêm instruction "Ignore length, focus on accuracy" vào prompt; normalize scores theo length |
| Style bias | — | Strip markdown trước khi pass vào judge |

**Cập nhật judge prompt sau phân tích:**

```
You are an impartial evaluator. Compare two answers to the same question.
IMPORTANT: Do NOT favor longer answers. Focus ONLY on:
- Factual accuracy
- Direct relevance to the question
- Clarity (not quantity)

Ignore formatting, length, and writing style.
...
```

---

## Kết luận

Judge hiện tại _(production-ready / cần cải thiện)_ với kappa = X.XX.

Điểm mạnh: swap-and-average hiệu quả giảm position bias (tie rate tăng X%).
Điểm yếu: _(ví dụ: length bias còn rõ ở pairs có chênh lệch lớn)_.

Recommendation: _(ví dụ: "Dùng judge này cho monitoring/alerting, nhưng không dùng làm ground truth
cuối cùng. Human review ít nhất 10% samples mỗi tuần để track drift.")_
