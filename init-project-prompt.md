<!--
  init-project-prompt.md
  ========================
  Prompt khởi tạo Second Brain cho dự án mới.
  Hoạt động với MỌI AI agent: Claude Code, Copilot Chat, Cursor, ChatGPT, Gemini.

  Cách dùng:
  1. Xem bảng "Chọn agent" bên dưới để biết agent bạn đang dùng cần làm gì
  2. Copy prompt tương ứng, điền [PROJECT_NAME] và [PROJECT_PATH]
  3. Chạy — agent tự đọc _global/ và tạo tất cả files
-->

---

## Chọn agent của bạn

| Agent | Đọc file được? | Config tự động load | Dùng prompt nào |
|-------|---------------|--------------------|----|
| **Claude Code** (terminal) | ✅ trực tiếp | `CLAUDE.md` | [→ Prompt A](#prompt-a--claude-code) |
| **Copilot Chat** (VS Code) | ✅ qua `@workspace` | `.github/copilot-instructions.md` | [→ Prompt B](#prompt-b--copilot-chat) |
| **Cursor** | ✅ trực tiếp | `.cursor/rules` | [→ Prompt C](#prompt-c--cursor) |
| **ChatGPT / Gemini** (web) | ❌ không đọc file | Không có | [→ Prompt D](#prompt-d--chatgpt--gemini-web) |

---

## CORE PROMPT — Nhân chung cho mọi agent

Phần logic chính, được nhúng vào từng prompt bên dưới:

```
Tôi muốn khởi tạo Second Brain cho dự án [PROJECT_NAME].
Project nằm tại: [PROJECT_PATH]

Hãy thực hiện theo thứ tự sau:

## Bước 1 — Đọc global context của tôi
Đọc các file sau để hiểu preferences và kinh nghiệm của tôi:
- d:\SecondBrain\_global\my-stack.md
- d:\SecondBrain\_global\patterns.md
- d:\SecondBrain\_global\mistakes.md

## Bước 2 — Đọc codebase dự án
Đọc codebase tại [PROJECT_PATH]:
- Xem cấu trúc thư mục (2 level)
- Đọc package.json / requirements.txt / go.mod (tuỳ stack)
- Đọc README.md nếu có
- Đọc các file config chính (tsconfig, .env.example, docker-compose, v.v.)
- Đọc 2-3 file source quan trọng nhất để hiểu architecture

## Bước 3 — Tạo Second Brain

Tạo thư mục: d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\

Tạo d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\CONTEXT.md:
- Điền đầy đủ: tổng quan, tech stack thực tế từ codebase, cấu trúc thư mục,
  trạng thái hiện tại (đoán từ code), key patterns đang dùng
- Phần TODO: để trống, hỏi tôi sau
- Dựa trên my-stack.md để biết conventions tôi ưa thích

Tạo d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\DECISIONS.md:
# Decisions — [PROJECT_NAME]
Lịch sử quyết định kiến trúc. Chỉ append, không xóa.
---
### [ngày hôm nay] Khởi tạo — Stack được chọn: [điền từ codebase]

Tạo d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\MISTAKES.md:
# Mistakes — [PROJECT_NAME]
Lỗi đã gặp + cách fix. Chỉ append, không xóa.
---
(trống, sẵn sàng nhận entries)

Tạo thư mục trống: d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\sessions\

## Bước 4 — Báo cáo

Sau khi xong, báo cáo:
- Danh sách files đã tạo
- Tóm tắt những gì đọc được từ codebase
- 3 câu hỏi để tôi điền phần TODO trong CONTEXT.md
```

---

## Prompt A — Claude Code

> Dùng trong terminal Claude Code (`claude` command). Claude Code đọc file hệ thống trực tiếp.

```
[Paste CORE PROMPT ở trên, điền PROJECT_NAME và PROJECT_PATH]

## Bước 5 — Tạo CLAUDE.md

Tạo [PROJECT_PATH]\CLAUDE.md dựa trên template:
d:\SecondBrain\AI Agentic\wiki\templates\CLAUDE-md-template.md
Điền đầy đủ từ Bước 2. Không để [placeholder] còn lại.

## Bước 6 — Tạo copilot-instructions

Tạo [PROJECT_PATH]\.github\copilot-instructions.md dựa trên template:
d:\SecondBrain\AI Agentic\wiki\templates\copilot-instructions-template.md
Điền đầy đủ, giữ dưới 300 dòng.
```

**Session tiếp theo với Claude Code:**
```
# Start
Đọc d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\CONTEXT.md — tóm tắt trạng thái.

# End
Kết thúc session theo đúng format sau:

1) Cập nhật d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\CONTEXT.md với 4 mục:
- Đã làm
- Files thay đổi
- Quyết định
- TODO tiếp theo

2) Tạo file log:
d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\sessions\[YYYY-MM-DD]-[short-slug].md

3) Nội dung session log theo template:
# Session Log — [YYYY-MM-DD] [PROJECT_NAME]
## Mục tiêu
## Đã làm
## Files thay đổi
## Quyết định
## Rủi ro / nợ kỹ thuật
## TODO phiên sau

4) Trả về checklist xác nhận:
- [ ] CONTEXT.md đã cập nhật
- [ ] Session log đã tạo
- [ ] TODO phiên sau đã rõ
```

---

## Prompt B — Copilot Chat

> Dùng trong VS Code Copilot Chat. Copilot đọc file qua `@workspace`, tự động load `.github/copilot-instructions.md`.

```
@workspace 

[Paste CORE PROMPT ở trên, điền PROJECT_NAME và PROJECT_PATH]

Lưu ý: Khi đọc file _global/, hãy dùng đường dẫn đầy đủ:
d:\SecondBrain\_global\my-stack.md

## Bước 5 — Tạo copilot-instructions

Tạo .github/copilot-instructions.md trong workspace hiện tại.
Dựa trên template: d:\SecondBrain\AI Agentic\wiki\templates\copilot-instructions-template.md
Điền từ thông tin đọc được ở Bước 2. Giữ dưới 300 dòng.
File này sẽ được Copilot tự load mọi session sau.
```

**Session tiếp theo với Copilot:**
```
# Start (Copilot đã tự load copilot-instructions.md)
@workspace Đọc d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\CONTEXT.md
Tóm tắt trạng thái dự án hiện tại.

# End
@workspace Kết thúc session hôm nay.
Cập nhật d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\CONTEXT.md theo cấu trúc:
- Đã làm / Files thay đổi / Quyết định / TODO tiếp theo

Tạo session log tại:
d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\sessions\[YYYY-MM-DD]-[short-slug].md

Cuối cùng trả về checklist:
- [ ] CONTEXT.md đã cập nhật
- [ ] Session log đã tạo
- [ ] TODO phiên sau đã rõ
```

---

## Prompt C — Cursor

> Dùng trong Cursor Chat. Cursor đọc file trực tiếp, tự động load `.cursor/rules`.

```
[Paste CORE PROMPT ở trên, điền PROJECT_NAME và PROJECT_PATH]

## Bước 5 — Tạo Cursor rules

Tạo .cursor/rules trong project root.
Nội dung: copy từ MEMORY\CONTEXT.md (phần tech stack + patterns + quy tắc).
Giữ dưới 200 dòng — đây là context Cursor load tự động.

## Bước 6 — Tạo copilot-instructions (nếu cũng dùng Copilot)

Tạo .github/copilot-instructions.md dựa trên template:
d:\SecondBrain\AI Agentic\wiki\templates\copilot-instructions-template.md
```

**Session tiếp theo với Cursor:**
```
# Start
Đọc d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\CONTEXT.md
Tóm tắt trạng thái dự án.

# End
Kết thúc session theo format:
- Cập nhật CONTEXT.md: Đã làm / Files thay đổi / Quyết định / TODO tiếp theo
- Tạo MEMORY\sessions\[YYYY-MM-DD]-[short-slug].md
- Trả về checklist xác nhận 3 mục đã hoàn tất
```

---

## Prompt D — ChatGPT / Gemini (web)

> Không có file access. Bạn phải paste nội dung thủ công. Dùng cách sau để tốn ít công nhất.

**Khởi tạo — paste 1 lần:**
```
Tôi muốn xây Second Brain cho dự án [PROJECT_NAME].

Đây là global preferences của tôi:
---
[Paste nội dung d:\SecondBrain\_global\my-stack.md vào đây]
---

Đây là mô tả dự án:
- Tên: [PROJECT_NAME]
- Tech stack: [mô tả ngắn]
- Cấu trúc chính: [mô tả ngắn]
- Trạng thái hiện tại: [đang làm gì]

Hãy tạo cho tôi nội dung của các files sau (xuất ra text để tôi copy-paste):
1. MEMORY\CONTEXT.md
2. MEMORY\DECISIONS.md (chỉ header + entry đầu tiên)
3. MEMORY\MISTAKES.md (chỉ header)

Dựa trên my-stack.md để biết conventions tôi ưa thích.
```

**Sau đó:** Copy output → tự tạo files vào `d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\`

**Session tiếp theo với ChatGPT/Gemini:**
```
# Start — paste mỗi session
Đây là context dự án [PROJECT_NAME] của tôi:
---
[Paste nội dung MEMORY\CONTEXT.md vào đây]
---
Tiếp tục từ đây.

# End — yêu cầu update
Tóm tắt những gì vừa làm để tôi cập nhật MEMORY\CONTEXT.md.
Format bắt buộc: Đã làm / Files thay đổi / Quyết định / TODO tiếp theo.
Đồng thời xuất thêm 1 block "Session Log" để tôi copy vào file sessions/[YYYY-MM-DD]-[short-slug].md.
```

---

## Sau khi khởi tạo xong (mọi agent)

Bạn chỉ cần:
1. Xác nhận báo cáo — sửa nếu AI hiểu sai
2. Điền 3 câu hỏi TODO mà AI hỏi
3. Điền `my-stack.md` nếu chưa điền (chỉ lần đầu)

## Các session tiếp theo

**Start:**
```
Load d:\SecondBrain\_projects\[PROJECT_NAME]\MEMORY\CONTEXT.md — tóm tắt trạng thái hiện tại.
```

**End:**
```
Kết thúc session theo protocol:
1) Cập nhật CONTEXT.md (Đã làm / Files thay đổi / Quyết định / TODO tiếp theo)
2) Tạo sessions/[YYYY-MM-DD]-[short-slug].md
3) Trả checklist xác nhận 3 mục đã xong
```
