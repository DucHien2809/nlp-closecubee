# Selected examples for demo and error analysis

Mỗi mẫu có SRC/REF và dự đoán của ViT5-base, BARTpho, MBR, FELIX++ để so sánh trên slide.

## 1. correct - deletion_only - id=76

**Nhận xét gợi ý:** Đúng: xóa được từ chức năng/thừa.

- SRC: `Tên ký hiệu của bạn là gì ?`
- REF: `Tên ký hiệu bạn gì ?`
- ViT5-base: `Tên ký hiệu bạn gì ?`
- BARTpho: `Tên ký hiệu bạn gì ?`
- MBR: `Tên ký hiệu bạn gì ?`
- FELIX++: `Tên ký hiệu bạn gì ?`

## 2. correct - reorder_only - id=67

**Nhận xét gợi ý:** Đúng: học được đảo trật tự gloss.

- SRC: `Tôi thích nhất màu trắng .`
- REF: `Tôi màu trắng thích nhất .`
- ViT5-base: `Tôi màu trắng thích nhất .`
- BARTpho: `Tôi thích nhất màu trắng .`
- MBR: `Tôi màu trắng thích nhất .`
- FELIX++: `Tôi màu trắng thích nhất .`

## 3. correct - deletion_reorder - id=28

**Nhận xét gợi ý:** Đúng: kết hợp xóa từ và đổi trật tự.

- SRC: `1 năm có mấy tháng ?`
- REF: `1 năm tháng mấy ?`
- ViT5-base: `1 năm tháng mấy ?`
- BARTpho: `1 năm tháng mấy ?`
- MBR: `1 năm tháng mấy ?`
- FELIX++: `1 năm tháng mấy ?`

## 4. correct - identical - id=14

**Nhận xét gợi ý:** Đúng: câu gần copy được giữ nguyên.

- SRC: `Bạn học lớp mấy ?`
- REF: `Bạn học lớp mấy ?`
- ViT5-base: `Bạn học lớp mấy ?`
- BARTpho: `Bạn học lớp mấy ?`
- MBR: `Bạn học lớp mấy ?`
- FELIX++: `Bạn học lớp ?`

## 5. mbr_improvement - identical - id=548

**Nhận xét gợi ý:** MBR chọn đúng dù ít nhất một backbone sai.

- SRC: `Tôi có hai con mèo .`
- REF: `Tôi có hai con mèo .`
- ViT5-base: `Tôi có hai con mèo .`
- BARTpho: `Tôi mèo hai .`
- MBR: `Tôi có hai con mèo .`
- FELIX++: `Tôi mèo có hai ?`

## 6. mbr_improvement - deletion_only - id=606

**Nhận xét gợi ý:** MBR chọn đúng dù ít nhất một backbone sai.

- SRC: `Nam đã đạt được một thỏa thuận .`
- REF: `Nam đạt được một thỏa thuận .`
- ViT5-base: `Nam đạt được một thỏa thuận .`
- BARTpho: `Nam đạt được một thuận .`
- MBR: `Nam đạt được một thỏa thuận .`
- FELIX++: `Nam đạt được một thỏa ?`

## 7. mbr_improvement - deletion_only - id=1855

**Nhận xét gợi ý:** MBR chọn đúng dù ít nhất một backbone sai.

- SRC: `Khi nào thì Nam cần nó ?`
- REF: `Khi Nam nó ?`
- ViT5-base: `Khi Nam nó ?`
- BARTpho: `Khi Nam cần nó ?`
- MBR: `Khi Nam nó ?`
- FELIX++: `khi Nam cần nó ?`

## 8. felix_correct - deletion_reorder - id=19

**Nhận xét gợi ý:** FELIX++ đúng: vừa xóa vừa đổi trật tự.

- SRC: `1 tuần có 7 ngày .`
- REF: `1 tuần ngày 7 .`
- ViT5-base: `1 tuần ngày 7 ngày .`
- BARTpho: `1 tuần ngày 7.`
- MBR: `1 tuần ngày 7 ngày .`
- FELIX++: `1 tuần ngày 7 .`

## 9. error - reorder_only - id=11

**Nhận xét gợi ý:** Sai: khó đảo trật tự chính xác.

- SRC: `Con gà ăn thóc .`
- REF: `Con gà thóc ăn .`
- ViT5-base: `Con gà ăn thóc .`
- BARTpho: `Con gà ăn thóc .`
- MBR: `Con gà ăn thóc .`
- FELIX++: `Con gà ăn ?`

## 10. error - deletion_only - id=114

**Nhận xét gợi ý:** Sai: xóa thiếu/thừa token.

- SRC: `Hôm qua là chủ nhật ngày 21 tháng 7 năm 2013`
- REF: `Hôm qua chủ nhật ngày 21 tháng 7 năm 2013`
- ViT5-base: `Hôm qua chủ nhật ngày 21 tháng 7 năm 2013.`
- BARTpho: `Hôm qua chủ nhật ngày 21 tháng 7 năm 2013.`
- MBR: `Hôm qua chủ nhật ngày 21 tháng 7 năm 2013.`
- FELIX++: `Hôm qua chủ nhật ngày 21 tháng 7 năm 2013 ?`

## 11. error - lexical - id=44

**Nhận xét gợi ý:** Sai: biến đổi lexical/cụm hóa khó với dữ liệu ít.

- SRC: `Nhà bạn có mấy người ?`
- REF: `Gia đình bạn người mấy ?`
- ViT5-base: `Nhà bạn người mấy ?`
- BARTpho: `Nhà bạn người mấy ?`
- MBR: `Nhà bạn người mấy ?`
- FELIX++: `Nhà bạn người mấy ?`

## 12. felix_error - lexical - id=153

**Nhận xét gợi ý:** FELIX++ sai trong khi MBR đúng; hữu ích để phân tích hạn chế edit-based hiện tại.

- SRC: `Nhanh lên nào !`
- REF: `Nhanh lên .`
- ViT5-base: `Nhanh lên .`
- BARTpho: `Nhanh lên .`
- MBR: `Nhanh lên .`
- FELIX++: `Nhanh lên ?`

## 13. felix_error - identical - id=206

**Nhận xét gợi ý:** FELIX++ sai trong khi MBR đúng; hữu ích để phân tích hạn chế edit-based hiện tại.

- SRC: `Anh phải lòng em .`
- REF: `Anh phải lòng em .`
- ViT5-base: `Anh phải lòng em .`
- BARTpho: `Anh phải lòng em .`
- MBR: `Anh phải lòng em .`
- FELIX++: `Anh phải lòng em ?`

## 14. felix_error - identical - id=216

**Nhận xét gợi ý:** FELIX++ sai trong khi MBR đúng; hữu ích để phân tích hạn chế edit-based hiện tại.

- SRC: `Hết thời gian .`
- REF: `Hết thời gian .`
- ViT5-base: `Hết thời gian .`
- BARTpho: `Hết thời gian .`
- MBR: `Hết thời gian .`
- FELIX++: `Hết thời gian ?`
