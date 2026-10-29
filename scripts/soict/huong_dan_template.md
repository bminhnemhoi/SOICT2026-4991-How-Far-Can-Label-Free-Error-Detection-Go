# Hướng dẫn nộp camera-ready — SOICT 2026, Paper 4991

**Bài:** *How Far Can Label-Free Error Detection Go? Evidential Fusion of Data Profilers and LLM Judges for Tables*
**Tác giả liên hệ (duy nhất):** Binh Minh Ngo — ngobinhminh.st@tdtu.edu.vn
**Hạn chót:** **23:59 ngày 30/10/2026** cho cả camera-ready lẫn đăng ký và đóng phí. Thiếu một trong hai thì bài không vào kỷ yếu.

## 1. Ba file cần upload (trong thư mục `1_UPLOAD_EASYCHAIR/`)

| Ô trên trang EasyChair "Update or Upload Files for Paper 4991" | File |
|---|---|
| **Zip file** | `4991.zip` ({zip_kb} KB, SHA-256 bắt đầu `{zip_sha}`) |
| **PDF file** | `4991.pdf` ({pages} trang, {pdf_kb} KB, SHA-256 bắt đầu `{pdf_sha}`) |
| **Signed copyright form** | `4991_Licence_to_Publish_signed.pdf` |

`4991.zip` đáp ứng cả hai yêu cầu:
- **EasyChair:** file `main.tex` nằm ở **thư mục gốc** của ZIP, vì EasyChair biên dịch từ đó.
- **SOICT:** có **thư mục `Source/`** chứa đủ mã nguồn, kèm `Corresponding_Author.txt` và Licence đã ký.

Đã mô phỏng EasyChair (`pdflatex` → `bibtex` → `pdflatex` ×2) cả ở thư mục gốc lẫn trong `Source/`: 0 lỗi, {pages} trang, cho ra đúng file PDF nộp.

Danh sách file trong ZIP:
{listing}

## 2. Điền form EasyChair từng ô (theo ảnh chụp)

1. Mở https://easychair.org/conferences/?conf=soict2026 → **My Papers** → bài **4991** → **Update or Upload Files**.
2. Dùng **khung bên trái "Paper Prepared Using LaTeX"**. **Không** điền khung "Microsoft Word" bên phải.
3. Điền các ô:
   - **Zip file:** bấm *Chọn tệp* → `1_UPLOAD_EASYCHAIR/4991.zip`
   - **PDF file:** bấm *Chọn tệp* → `1_UPLOAD_EASYCHAIR/4991.pdf`
   - **Signed copyright form:** bấm *Chọn tệp* → `1_UPLOAD_EASYCHAIR/4991_Licence_to_Publish_signed.pdf`
   - **Name of the main LaTeX file:** gõ đúng `main.tex`
   - **Program to process the main file:** `pdflatex` (giữ nguyên mặc định)
   - **Program to process the bibliography:** `bibtex` (giữ nguyên mặc định)
4. Bấm nút đỏ **Submit a New Version** ở khung bên trái.
5. Nếu EasyChair báo lỗi biên dịch, chụp màn hình gửi lại để xử lý. Không tự sửa file.
6. **Kiểm tra lại:** vào lại trang bài 4991, tải từng file về mở thử. PDF phải {pages} trang, có dấu ✉ cạnh tên.
7. Sửa thông tin tác giả cho khớp với bài:
   - tên là **Binh Minh Ngo** (*First name* Binh Minh, *Last name* Ngo);
   - email là **ngobinhminh.st@tdtu.edu.vn**.

   Nếu EasyChair hỏi tóm tắt hoặc từ khoá, dán từ `2_THONG_TIN_EASYCHAIR/EasyChair_metadata.txt`.
   - ⚠️ EasyChair đang ghi **"Bình Minh Ngô"** (có dấu). Springer lấy tên từ EasyChair, nên hãy sửa cho khớp với bài.
   - ⚠️ Licence dùng **ảnh chữ ký**, trong khi quy định chính thức ghi *"Digital signatures are not accepted"*. Bạn đã xác nhận ban tổ chức đồng ý. Nếu họ vẫn yêu cầu ký tay: in `3_HO_SO_LUU/4991_Licence_to_Publish_prefilled_PRINT.pdf`, ký bằng mực ở trang 5, scan cả 6 trang thành PDF, rồi upload lại vào ô *Signed copyright form*.

## 3. Đăng ký và đóng phí (bắt buộc, cùng hạn 30/10)

- Link (form tính phí bằng VND của ban thư ký): https://soictvn2026.websitehoinghi.com/registration
- Chọn loại **Full registration**, mức có tài trợ **7.000.000 VND**. Vì bạn là tác giả duy nhất, mức miễn phí cho "student co-author" không áp dụng.
- Ghi **Paper ID 4991** khi đăng ký, đóng phí, rồi **lưu biên lai hoặc ảnh chụp giao dịch**.

## 4. Danh sách kiểm tra trước khi bấm nộp

- [ ] Upload đúng 2 file trong `1_UPLOAD_EASYCHAIR/`, không lấy bản cũ ở nơi khác.
- [ ] PDF mở được, {pages} trang, đúng tiêu đề và tên tác giả.
- [ ] Email tác giả liên hệ giống nhau ở 4 nơi: bài báo, Licence, file txt, EasyChair (`ngobinhminh.st@tdtu.edu.vn`).
- [ ] Đã đăng ký và đóng phí, có biên lai.
- [ ] Đã mở lại EasyChair để xác nhận file đã lên.
- [ ] Sau khi nộp: thu hồi mọi API key và key router đã dán trong các cuộc trò chuyện.

## 5. Nếu gặp sự cố

Gửi email cho ban tổ chức: **contact@soict.hust.edu.vn**, kèm Paper ID 4991, tên bài, email tài khoản EasyChair, và ảnh chụp lỗi.

Khi bài được xử lý xong, Springer sẽ gửi bản proof cho tác giả liên hệ để kiểm tra. Hãy theo dõi email `ngobinhminh.st@tdtu.edu.vn` và phản hồi nhanh.

## 6. Sau hạn nộp

- Hội nghị diễn ra **4–5/12/2026** tại TP.HCM. Phần trình bày oral dài **15 phút, đã gồm Q&A**. Có mặt sớm 5–10 phút để chép slide và báo session chair.
- Poster là tuỳ chọn: gửi file cho ban tổ chức, họ in miễn phí.
- Code và kết quả công khai tại: https://github.com/bminhnemhoi/SOICT2026-4991-How-Far-Can-Label-Free-Error-Detection-Go
