# Báo cáo bài tập xử lý ảnh: Tách các ô có ghi nước đi viết tay trên phiếu ghi ván cờ

## 1. Giới thiệu bài toán

Bài toán được chọn là xử lý ảnh phiếu ghi lịch sử các nước đi của một ván cờ.

Ảnh đầu vào là ảnh chụp hoặc ảnh scan của một bảng ghi nước đi. Trong bảng này, các nước đi được viết tay trong từng ô.

Yêu cầu của bài toán là xây dựng một chuỗi xử lý ảnh duy nhất để áp dụng cho tất cả các ảnh đầu vào. Chương trình không được chỉnh tham số thủ công theo từng ảnh, mà phải dùng cùng một pipeline xử lý với cùng bộ tham số.

Input:

- Một ảnh phiếu ghi nước đi cờ vua.

Output:

- Khoanh vùng hoặc tách ra các ô có khả năng chứa nội dung viết tay.
- Kết quả được lưu thành các ảnh con tương ứng với từng ô.

---

## 2. Mục tiêu xử lý

Mục tiêu của chương trình là:

- Phát hiện cấu trúc bảng trong ảnh.
- Tách các đường kẻ ngang và dọc của bảng.
- Xác định các ô hình chữ nhật tương ứng với vùng ghi nước đi.
- Cắt từng ô ra thành ảnh riêng.
- Loại bỏ bớt phần số thứ tự nước đi in sẵn nếu cần.
- Có thể lọc và chỉ giữ lại các ô có dấu hiệu chứa chữ viết tay.

Bài toán này không yêu cầu nhận dạng nội dung nước đi, mà chỉ tập trung vào việc phát hiện và tách đúng vùng ô có chữ viết tay.

---

## 3. Chuỗi xử lý ảnh đề xuất

Chuỗi xử lý ảnh được sử dụng gồm các bước chính sau:

1. Đọc ảnh đầu vào.
2. Chuyển ảnh sang ảnh xám.
3. Nhị phân hóa ảnh bằng phương pháp Otsu.
4. Trích xuất đường kẻ bảng bằng phép toán hình thái học.
5. Tìm contour các vùng hình chữ nhật.
6. Lọc các contour có kích thước phù hợp.
7. Sắp xếp các ô theo đúng thứ tự trên phiếu ghi cờ.
8. Cắt từng ô từ ảnh gốc.
9. Cắt bỏ phần số thứ tự nước đi in sẵn ở một số cột.
10. Tùy chọn lọc các ô có chữ viết tay.
11. Lưu kết quả ra thư mục đầu ra.

Toàn bộ chuỗi xử lý trên được áp dụng giống nhau cho tất cả các ảnh.

---

## 4. Mô tả chi tiết từng bước

## 4.1. Đọc ảnh và chuyển sang ảnh xám

### Phương pháp

Ảnh đầu vào được đọc bằng thư viện PIL. Sau đó ảnh được chuyển sang hai dạng:

- Ảnh RGB.
- Ảnh mức xám.

Trong chương trình, bước này tương ứng với các lệnh:

- `Image.open(image_path).convert("RGB")`
- `image.convert("L")`

### Tham số

- Không sử dụng tham số đặc biệt.
- Ảnh được đưa về dạng RGB trước, sau đó chuyển sang ảnh xám.

### Lý do lựa chọn

Bài toán chủ yếu dựa vào sự khác biệt giữa nền giấy, đường kẻ bảng và chữ viết tay. Các đối tượng này khác nhau chủ yếu về cường độ sáng tối, không phụ thuộc nhiều vào màu sắc.

Vì vậy, chuyển sang ảnh xám giúp giảm độ phức tạp của dữ liệu, đồng thời làm cho các bước nhị phân hóa và trích xuất đường kẻ dễ thực hiện hơn.

### Ảnh hưởng đến kết quả

Việc chuyển ảnh sang mức xám giúp pipeline xử lý ổn định hơn. Tuy nhiên, nếu ảnh có ánh sáng không đều, bị bóng hoặc bị mờ, mức xám của nền và chữ có thể không tách biệt rõ, ảnh hưởng đến bước nhị phân hóa phía sau.

---

## 4.2. Nhị phân hóa ảnh bằng phương pháp Otsu

### Phương pháp

Ảnh xám được chuyển thành ảnh nhị phân bằng phương pháp Otsu.

Trong chương trình, bước này được thực hiện bởi hàm `cv2.threshold` với kiểu ngưỡng:

- `cv2.THRESH_BINARY`
- `cv2.THRESH_OTSU`

### Tham số

Các tham số chính:

- Ngưỡng ban đầu: `0`
- Giá trị điểm ảnh tối đa: `255`
- Phương pháp chọn ngưỡng: Otsu
- Kiểu ảnh đầu ra: ảnh nhị phân trắng đen

### Lý do lựa chọn

Otsu là phương pháp chọn ngưỡng tự động. Điều này phù hợp với yêu cầu của bài toán vì chương trình không được phép để người dùng chỉnh ngưỡng thủ công cho từng ảnh.

Ảnh phiếu ghi cờ thường có hai nhóm điểm ảnh chính:

- Nhóm sáng: nền giấy.
- Nhóm tối: đường kẻ bảng và chữ viết tay.

Do đó, Otsu có thể tự động tìm một ngưỡng phù hợp để tách hai nhóm này.

### Ảnh hưởng đến kết quả trung gian

Sau bước này, ảnh được đưa về dạng nhị phân:

- Nền giấy có màu trắng.
- Đường kẻ bảng và chữ viết tay có màu đen.

Kết quả nhị phân là đầu vào quan trọng cho bước trích xuất đường kẻ bảng.

### Hạn chế

Nếu ảnh bị chụp trong điều kiện ánh sáng không đều, bị bóng hoặc nền giấy quá tối, Otsu có thể chọn ngưỡng chưa tốt. Khi đó:

- Một số đường kẻ có thể bị đứt.
- Một số vùng nền có thể bị nhầm là nét tối.
- Chữ viết tay nhạt có thể bị mất một phần.

---

## 4.3. Trích xuất đường kẻ bảng bằng phép toán hình thái học

### Phương pháp

Sau khi nhị phân hóa, ảnh được đảo màu để đường kẻ và chữ viết tay trở thành vùng trắng trên nền đen.

Tiếp theo, chương trình tạo hai kernel hình chữ nhật:

- Kernel ngang để phát hiện đường kẻ ngang.
- Kernel dọc để phát hiện đường kẻ dọc.

Đường ngang được trích xuất bằng phép erosion rồi dilation với kernel ngang.

Đường dọc được trích xuất bằng phép erosion rồi dilation với kernel dọc.

Cuối cùng, ảnh đường kẻ bảng được tạo bằng cách cộng ảnh đường ngang và ảnh đường dọc.

### Tham số

Các tham số chính:

- `horizontal_kernel_divisor = 40`
- `vertical_kernel_divisor = 40`
- Chiều dài kernel ngang: `image_width // 40`
- Chiều dài kernel dọc: `image_height // 40`
- Số lần erosion: `1`
- Số lần dilation: `1`

### Lý do lựa chọn

Bảng ghi nước đi có cấu trúc gồm các đường kẻ ngang và dọc rõ ràng. Các đường này thường dài hơn nhiều so với nét chữ viết tay.

Vì vậy:

- Kernel ngang dài giúp giữ lại các đường ngang và loại bỏ phần lớn nét chữ ngắn.
- Kernel dọc dài giúp giữ lại các đường dọc của bảng.
- Kích thước kernel được tính theo tỉ lệ kích thước ảnh nên có khả năng thích ứng với ảnh có độ phân giải khác nhau.

Tham số chia cho `40` là lựa chọn cân bằng. Nếu kernel quá ngắn, nhiều nét chữ viết tay có thể bị giữ lại. Nếu kernel quá dài, các đường kẻ bị đứt hoặc ngắn có thể bị mất.

### Ảnh hưởng đến kết quả trung gian

Kết quả của bước này là ảnh chỉ còn lại chủ yếu các đường kẻ của bảng.

Kết quả tốt khi:

- Đường ngang và đường dọc rõ ràng.
- Các ô tạo thành hình chữ nhật khép kín.
- Phần lớn chữ viết tay không còn xuất hiện trong ảnh đường kẻ.

Kết quả chưa tốt khi:

- Đường kẻ bị đứt.
- Một số nét chữ viết tay còn sót lại.
- Một số ô không khép kín nên khó tìm contour chính xác.

---

## 4.4. Tìm contour các ô hình chữ nhật

### Phương pháp

Từ ảnh đường kẻ bảng, chương trình sử dụng `cv2.findContours` để tìm các contour.

Sau đó, mỗi contour được xấp xỉ thành đa giác bằng `cv2.approxPolyDP`.

Chỉ các contour có 4 đỉnh mới được giữ lại vì ô trong bảng có dạng gần giống hình chữ nhật.

### Tham số

Các tham số chính:

- Chế độ tìm contour: `cv2.RETR_LIST`
- Kiểu nén contour: `cv2.CHAIN_APPROX_SIMPLE`
- Hệ số xấp xỉ contour: `0.04 * perimeter`
- Điều kiện số đỉnh: `len(approx) == 4`
- Diện tích contour phải lớn hơn `0`

### Lý do lựa chọn

Mỗi ô ghi nước đi trong bảng có dạng hình chữ nhật. Vì vậy, sau khi trích xuất đường kẻ bảng, contour của các ô hợp lệ thường có thể được xấp xỉ thành đa giác 4 đỉnh.

Việc dùng `approxPolyDP` giúp đơn giản hóa contour, loại bỏ các dao động nhỏ trên biên và giúp kiểm tra hình dạng dễ hơn.

### Ảnh hưởng đến kết quả

Bước này tạo ra danh sách các ứng viên có thể là ô ghi nước đi.

Các vùng không phải hình chữ nhật, ví dụ nhiễu, nét chữ viết tay hoặc vùng nhỏ bất thường, thường bị loại bỏ.

### Hạn chế

Nếu ảnh bị nghiêng, méo phối cảnh hoặc đường kẻ bảng bị đứt, contour của ô có thể không còn là hình chữ nhật khép kín. Khi đó, chương trình có thể bỏ sót một số ô hợp lệ.

---

## 4.5. Lọc contour theo diện tích

### Phương pháp

Sau khi tìm được các contour hình chữ nhật, chương trình tính diện tích của từng contour. Sau đó lấy trung vị diện tích của toàn bộ các contour.

Chỉ các contour có diện tích nằm trong khoảng từ `0.5 * median_area` đến `1.5 * median_area` được giữ lại.

### Tham số

Các tham số chính:

- Cận dưới diện tích: `0.5 * median_area`
- Cận trên diện tích: `1.5 * median_area`

### Lý do lựa chọn

Các ô ghi nước đi trên cùng một phiếu thường có kích thước gần giống nhau. Trong khi đó, các contour nhiễu hoặc contour không phải ô thường có diện tích quá nhỏ hoặc quá lớn.

Sử dụng trung vị thay vì trung bình giúp kết quả ổn định hơn, vì trung vị ít bị ảnh hưởng bởi các contour bất thường.

### Ảnh hưởng đến kết quả

Bước này giúp loại bỏ:

- Vùng nhiễu nhỏ.
- Vùng bao quanh cả bảng.
- Vùng không phải ô ghi nước đi.
- Contour bất thường do nét chữ hoặc đường kẻ ngoài bảng.

### Hạn chế

Nếu nhiều ô bị phát hiện sai hoặc nếu bảng có các ô kích thước không đồng đều, trung vị diện tích có thể không còn đại diện tốt. Khi đó, một số ô hợp lệ có thể bị loại nhầm.

---

## 4.6. Sắp xếp các ô theo thứ tự trên phiếu ghi cờ

### Phương pháp

Với mỗi contour, chương trình tính bounding box và tâm của ô:

- Tọa độ tâm theo chiều ngang: `x_center`
- Tọa độ tâm theo chiều dọc: `y_center`
- Chiều cao ô: `height`

Sau đó, chương trình gán chỉ số cột cho từng ô bằng cách:

1. Sắp xếp các ô theo `x_center`.
2. Tính khoảng cách giữa các ô liên tiếp theo trục x.
3. Lấy 3 khoảng cách lớn nhất để chia các ô thành 4 nhóm cột.

Tiếp theo, chương trình gom các ô thành hàng dựa trên `y_center`.

Hai ô được xem là cùng hàng nếu chênh lệch tâm theo trục y không vượt quá:

- `max(median_height * 0.65, 10.0)`

Cuối cùng, các ô được sắp xếp theo thứ tự đọc của phiếu ghi cờ:

- Nửa trái của bảng trước: cột 0 và cột 1.
- Nửa phải của bảng sau: cột 2 và cột 3.
- Trong mỗi nửa, duyệt từ trên xuống dưới.
- Trong mỗi hàng, duyệt từ trái sang phải.

### Tham số

Các tham số chính:

- Số nhóm cột giả định: `4`
- Số khoảng cách lớn nhất dùng để tách cột: `3`
- Ngưỡng gom hàng: `max(median_height * 0.65, 10.0)`

### Lý do lựa chọn

Phiếu ghi nước đi cờ thường có cấu trúc gồm nhiều cột, trong đó mỗi hàng chứa các ô tương ứng với nước đi của quân trắng và quân đen.

Việc sắp xếp theo tọa độ giúp đảm bảo các ô được lưu theo thứ tự logic, thuận tiện cho việc kiểm tra hoặc xử lý nhận dạng ở bước sau.

### Ảnh hưởng đến kết quả

Nếu các ô được phát hiện đầy đủ, bước này giúp tên file đầu ra phản ánh đúng thứ tự nước đi trong ván cờ.

### Hạn chế

Cách sắp xếp này giả định bảng có 4 nhóm cột chính. Nếu ảnh đầu vào có bố cục khác, hoặc bị xoay nghiêng mạnh, bước chia cột có thể bị sai.

---

## 4.7. Cắt ảnh từng ô

### Phương pháp

Với mỗi contour đã được chọn, chương trình lấy bounding box của contour rồi cắt vùng tương ứng trên ảnh xám.

Khi cắt, chương trình thêm padding theo chiều dọc:

- Padding phía trên bằng `15%` chiều cao ô.
- Padding phía dưới bằng `25%` chiều cao ô.

### Tham số

Các tham số chính:

- `top_padding = int(height * 0.15)`
- `bottom_padding = int(height * 0.25)`
- Không thêm padding theo chiều ngang.

### Lý do lựa chọn

Chữ viết tay trong ô có thể nằm lệch lên hoặc lệch xuống so với vùng contour phát hiện. Nếu cắt đúng sát contour, một phần nét chữ có thể bị mất.

Vì vậy, thêm padding theo chiều dọc giúp giữ lại đầy đủ nội dung viết tay hơn.

Padding phía dưới lớn hơn phía trên vì một số nét chữ hoặc nét kéo có thể nằm gần đáy ô.

### Ảnh hưởng đến kết quả

Ảnh ô sau khi cắt có khả năng chứa đầy đủ chữ viết tay hơn. Điều này hữu ích nếu ảnh ô tiếp tục được sử dụng cho bước nhận dạng chữ viết tay.

### Hạn chế

Nếu padding quá lớn, ảnh cắt có thể chứa thêm một phần đường kẻ hoặc nội dung của hàng bên cạnh. Tuy nhiên, với mức `15%` phía trên và `25%` phía dưới, mức mở rộng vẫn tương đối hợp lý.

---

## 4.8. Cắt bỏ phần số thứ tự nước đi in sẵn

### Phương pháp

Một số ô có thể chứa thêm phần số thứ tự nước đi được in sẵn ở bên trái. Đây không phải nội dung viết tay cần quan tâm.

Chương trình cắt bỏ một phần bên trái của các ô thuộc cột 0 hoặc cột 2.

### Tham số

Tham số chính:

- `trim_number_column_ratio = 0.28`

Nghĩa là cắt bỏ `28%` chiều rộng bên trái của các ô thuộc cột 0 và cột 2.

### Lý do lựa chọn

Phần số thứ tự nước đi là nội dung in sẵn, không phải chữ viết tay. Nếu giữ lại phần này, ảnh ô có thể chứa cả số in và nước đi viết tay, gây nhiễu cho các bước xử lý sau.

Cắt bỏ `28%` phía trái là lựa chọn phù hợp vì phần số thứ tự thường nằm ở đầu ô, còn phần nước đi viết tay thường nằm phía bên phải.

### Ảnh hưởng đến kết quả

Bước này giúp ảnh ô đầu ra tập trung hơn vào nội dung viết tay.

### Hạn chế

Nếu người viết ghi chữ quá sát mép trái, việc cắt bỏ `28%` có thể làm mất một phần nét chữ. Ngược lại, nếu phần số in rộng hơn `28%`, một phần số in có thể vẫn còn sót lại.

---

## 4.9. Lọc các ô có chữ viết tay

### Phương pháp

Chương trình có tùy chọn `--non-empty-only` để chỉ lưu các ô có khả năng chứa chữ viết tay.

Với mỗi ảnh ô, chương trình xét vùng bên trong ô và bỏ qua phần rìa ảnh. Sau đó tính tỉ lệ điểm ảnh tối trong vùng này.

Một ô được xem là có chữ viết tay nếu tỉ lệ điểm ảnh tối lớn hơn hoặc bằng ngưỡng cho trước.

### Tham số

Các tham số chính:

- Vùng xét theo chiều cao: từ `18%` đến `88%`
- Vùng xét theo chiều rộng: từ `6%` đến `94%`
- Điểm ảnh tối được định nghĩa là điểm ảnh có giá trị nhỏ hơn `170`
- Tỉ lệ điểm ảnh tối tối thiểu: `min_dark_ratio = 0.012`

### Lý do lựa chọn

Chữ viết tay thường tạo ra các điểm ảnh tối bên trong ô. Trong khi đó, ô rỗng chủ yếu là nền sáng, chỉ có thể còn một ít đường kẻ ở biên.

Vì vậy, chương trình bỏ qua phần rìa ảnh và chỉ xét vùng bên trong ô để tránh nhầm đường kẻ bảng là chữ viết tay.

Ngưỡng `0.012` tương đối nhỏ, phù hợp với trường hợp chữ viết tay mảnh hoặc ít nét.

### Ảnh hưởng đến kết quả

Bước này giúp giảm số lượng ảnh ô đầu ra, chỉ giữ lại các ô có khả năng chứa thông tin cần quan tâm.

### Hạn chế

Nếu chữ viết tay quá nhạt, tỉ lệ điểm ảnh tối có thể nhỏ hơn ngưỡng và ô có thể bị loại nhầm. Ngược lại, nếu ô rỗng có nhiều vết bẩn, nhiễu hoặc đường kẻ còn sót lại, ô rỗng có thể bị giữ lại nhầm.

---

## 5. Kết quả trung gian cần quan sát

Chương trình có tùy chọn `--save-debug` để lưu các ảnh trung gian quan trọng vào thư mục debug.

Các ảnh trung gian gồm:

- Ảnh nhị phân.
- Ảnh đường kẻ bảng.

## 5.1. Ảnh nhị phân

Ảnh nhị phân cho thấy kết quả tách nền giấy khỏi các nét tối.

Kết quả tốt khi:

- Nền giấy gần như trắng.
- Đường kẻ bảng rõ.
- Chữ viết tay rõ.
- Ít nhiễu trên nền.

Kết quả chưa tốt khi:

- Nền giấy bị đen thành từng vùng do bóng sáng.
- Đường kẻ bị đứt.
- Chữ viết tay quá mờ.
- Nhiễu nền xuất hiện nhiều.

## 5.2. Ảnh đường kẻ bảng

Ảnh đường kẻ bảng là kết quả sau khi trích xuất các đường ngang và dọc.

Kết quả tốt khi:

- Đường ngang và đường dọc của bảng được giữ lại rõ ràng.
- Các ô tạo thành hình chữ nhật khép kín.
- Phần lớn chữ viết tay đã bị loại bỏ.

Kết quả chưa tốt khi:

- Đường kẻ bị đứt.
- Một số nét chữ viết tay còn sót lại.
- Một số ô không khép kín nên không tìm được contour đúng.

---

## 6. Lưu kết quả đầu ra

Sau khi tách ô, chương trình lưu từng ô vào thư mục riêng theo tên ảnh đầu vào.

Nếu phát hiện đúng `120` ô, chương trình đặt tên file theo dạng:

- `<image_name>_cell_001_move_01_white.png`
- `<image_name>_cell_002_move_01_black.png`
- ...

Nếu số ô phát hiện khác `120`, chương trình đặt tên đơn giản theo số thứ tự:

- `<image_name>_cell_001.png`
- `<image_name>_cell_002.png`
- ...

Cách đặt tên này giúp dễ kiểm tra thứ tự các ô sau khi tách.

---

## 7. Đánh giá chuỗi xử lý

## 7.1. Ưu điểm

Chuỗi xử lý có các ưu điểm sau:

- Sử dụng một pipeline duy nhất cho tất cả ảnh.
- Không cần chỉnh tham số thủ công theo từng ảnh.
- Dùng Otsu để tự động chọn ngưỡng nhị phân.
- Khai thác tốt cấu trúc bảng bằng phép toán hình thái học.
- Có bước lọc contour theo hình dạng và diện tích để loại nhiễu.
- Có bước sắp xếp ô theo thứ tự logic của phiếu ghi cờ.
- Có thể loại bỏ phần số thứ tự nước đi in sẵn.
- Có thể lọc các ô rỗng, chỉ giữ lại ô có dấu hiệu chứa chữ viết tay.

## 7.2. Hạn chế

Chuỗi xử lý vẫn còn một số hạn chế:

- Phụ thuộc vào việc đường kẻ bảng phải đủ rõ và tương đối khép kín.
- Nếu ảnh bị nghiêng hoặc méo phối cảnh mạnh, contour ô có thể không chính xác.
- Nếu ánh sáng không đều, bước Otsu có thể nhị phân hóa chưa tốt.
- Nếu chữ viết tay quá nhạt, bước lọc ô có chữ có thể bỏ sót.
- Nếu bảng có bố cục khác với giả định 4 cột chính, bước sắp xếp cột có thể sai.
- Chưa có bước hiệu chỉnh phối cảnh cho ảnh chụp nghiêng.

---

## 8. Nhận xét về mức độ đạt yêu cầu

Chuỗi xử lý đã đáp ứng được yêu cầu chính của bài toán là xây dựng một pipeline xử lý ảnh hoàn chỉnh để tách các ô ghi nước đi viết tay.

Pipeline sử dụng cùng một bộ tham số cho tất cả ảnh, gồm:

- Nhị phân hóa bằng Otsu.
- Trích xuất đường kẻ bằng kernel hình thái học theo tỉ lệ kích thước ảnh.
- Lọc contour theo số đỉnh và diện tích trung vị.
- Gom hàng, chia cột và sắp xếp ô theo tọa độ.
- Cắt ô với padding cố định theo tỉ lệ chiều cao.
- Cắt bỏ `28%` phía trái ở các cột có số nước đi in sẵn.
- Lọc ô có chữ bằng tỉ lệ điểm ảnh tối tối thiểu `0.012`.

Kết quả đạt tốt trong trường hợp ảnh đầu vào có bảng rõ, ít nghiêng, đường kẻ đầy đủ và chữ viết tay đủ đậm.

Trong các trường hợp ảnh bị nghiêng, ánh sáng không đều hoặc đường kẻ bị mờ, kết quả có thể chưa hoàn hảo do contour của các ô không còn rõ ràng.

Tuy nhiên, về mặt thiết kế chuỗi xử lý, phương pháp này hợp lý vì đã kết hợp đầy đủ các bước quan trọng của xử lý ảnh truyền thống:

- Tiền xử lý.
- Phân đoạn.
- Hậu xử lý.
- Gán nhãn và trích xuất vùng quan tâm.

---

## 9. Hướng cải tiến

Để cải thiện kết quả, có thể bổ sung một số hướng sau:

## 9.1. Hiệu chỉnh phối cảnh

Nếu ảnh chụp bị nghiêng, có thể phát hiện biên ngoài của bảng rồi dùng phép biến đổi phối cảnh để đưa bảng về dạng thẳng trước khi tách ô.

## 9.2. Cải thiện xử lý ánh sáng

Có thể dùng adaptive threshold hoặc cân bằng sáng cục bộ để xử lý tốt hơn các ảnh bị bóng hoặc ánh sáng không đều.

## 9.3. Nối đường kẻ bị đứt

Có thể thêm phép closing để nối các đoạn đường kẻ bị đứt trước khi tìm contour.

## 9.4. Lọc ô bằng mô hình học máy

Thay vì chỉ dùng tỉ lệ điểm ảnh tối, có thể huấn luyện một mô hình phân loại nhỏ để xác định ô có chữ viết tay hay không.

## 9.5. Chuẩn hóa ảnh ô sau khi cắt

Sau khi tách ô, có thể tiếp tục xử lý:

- Loại bỏ đường kẻ còn sót.
- Chuẩn hóa kích thước ảnh.
- Làm mảnh nét chữ.
- Tăng tương phản chữ viết tay.

Các bước này sẽ hữu ích nếu tiếp tục phát triển hệ thống nhận dạng nội dung nước đi.

---

## 10. Kết luận

Báo cáo đã trình bày một chuỗi xử lý ảnh duy nhất để tách các ô có ghi nước đi viết tay trên phiếu ghi ván cờ.

Phương pháp chính dựa trên việc nhị phân hóa ảnh, trích xuất đường kẻ bảng bằng phép toán hình thái học, tìm contour hình chữ nhật, lọc contour theo diện tích, sau đó cắt từng ô ra thành ảnh riêng.

Chuỗi xử lý này phù hợp với bài toán vì ảnh đầu vào có cấu trúc bảng rõ ràng. Mặc dù vẫn còn một số hạn chế với ảnh nghiêng, ảnh mờ hoặc ánh sáng không đều, phương pháp đã đáp ứng được yêu cầu thiết kế một pipeline xử lý ảnh hoàn chỉnh, sử dụng cùng một bộ tham số cho tất cả ảnh đầu vào.