# vit5_large_constrained — error examples (test)

## identical  (100 mismatched)

- SRC : `Tôi đi học .`
  REF : `Tôi đi học .`
  HYP : `Tôi đi học`
- SRC : `Anh phải lòng em .`
  REF : `Anh phải lòng em .`
  HYP : `lòng em .`
- SRC : `Tôi béo vãi .`
  REF : `Tôi béo vãi .`
  HYP : `Tôi béo .`
- SRC : `Đừng có mà đánh nhau .`
  REF : `Đừng có mà đánh nhau .`
  HYP : `Đừng có đánh nhau .`
- SRC : `Quét phòng tôi đi .`
  REF : `Quét phòng tôi đi .`
  HYP : `Quét phòng tôi .`
- SRC : `Bạn trông tái nhợt .`
  REF : `Bạn trông tái nhợt .`
  HYP : `Bạn trông tái .`
- SRC : `Bạn nhớ tôi không ?`
  REF : `Bạn nhớ tôi không ?`
  HYP : `Bạn nhớ tôi ?`
- SRC : `Tôi muốn một con dao .`
  REF : `Tôi muốn một con dao .`
  HYP : `Tôi muốn một con`
- SRC : `Nó ghét cà rốt .`
  REF : `Nó ghét cà rốt .`
  HYP : `Nó ghét`
- SRC : `Đặt cái hộp xuống .`
  REF : `Đặt cái hộp xuống .`
  HYP : `Đặt cái hộp .`
- SRC : `Chúng ta phải làm nó .`
  REF : `Chúng ta phải làm nó .`
  HYP : `Chúng phải làm nó .`
- SRC : `Đừng làm điều đó .`
  REF : `Đừng làm điều đó .`
  HYP : `Đừng làm điều .`
- SRC : `Họ lấy nhau rồi .`
  REF : `Họ lấy nhau rồi .`
  HYP : `Họ lấy .`
- SRC : `Tôi bảo Nam làm rồi .`
  REF : `Tôi bảo Nam làm rồi .`
  HYP : `Tôi Nam làm rồi .`
- SRC : `Tôi yêu căn nhà đó .`
  REF : `Tôi yêu căn nhà đó .`
  HYP : `Tôi yêu nhà đó .`

## reorder_only  (7 mismatched)

- SRC : `Con gà ăn thóc .`
  REF : `Con gà thóc ăn .`
  HYP : `Con gà ăn thóc .`
- SRC : `Ai thích nhìn thác nước ?`
  REF : `Thác nước nhìn thích ai ?`
  HYP : `thích nhìn thác nước Ai ?`
- SRC : `Tôi học 5 môn : toán , văn , lý , hóa , sinh .`
  REF : `Tôi học môn 5 : toán , văn , lý , hóa , sinh .`
  HYP : `Tôi học 5 môn toán văn lý hóa sinh .`
- SRC : `Mẹ nấu cơm ngon .`
  REF : `Mẹ cơm ngon nấu .`
  HYP : `Mẹ nấu cơm .`
- SRC : `Sao mày không lớn lên ?`
  REF : `Sao mày lớn lên không ?`
  HYP : `Sao mày lớn .`
- SRC : `Ai đấy ?`
  REF : `đấy ai ?`
  HYP : `đấy Ai ?`
- SRC : `Tôi không đi chơi cùng với mẹ .`
  REF : `Tôi đi chơi cùng với mẹ không .`
  HYP : `Tôi đi chơi cùng với mẹ không`

## deletion_only  (152 mismatched)

- SRC : `Tên ký hiệu của bạn là gì ?`
  REF : `Tên ký hiệu bạn gì ?`
  HYP : `Tên ký hiệu bạn gì`
- SRC : `Hôm qua là chủ nhật ngày 21 tháng 7 năm 2013`
  REF : `Hôm qua chủ nhật ngày 21 tháng 7 năm 2013`
  HYP : `Hôm qua chủ nhật ngày 21 tháng 7 năm qua .`
- SRC : `Tôi nhìn thấy bố rất buồn .`
  REF : `Tôi nhìn bố buồn .`
  HYP : `Tôi nhìn thấy bố buồn .`
- SRC : `Tôi bắt đầu được không ạ ?`
  REF : `Tôi bắt đầu được không ?`
  HYP : `Tôi bắt đầu không ?`
- SRC : `Cô ấy yêu tôi .`
  REF : `Cô yêu tôi .`
  HYP : `Cô yêu tôi`
- SRC : `Lại đây đi .`
  REF : `Lại đây .`
  HYP : `đây đi .`
- SRC : `Đó là một cách chơi chữ .`
  REF : `Đó một cách chơi chữ .`
  HYP : `Đó một cách chơi chữ`
- SRC : `Chúng tôi sẽ giấu nó .`
  REF : `Chúng tôi giấu nó .`
  HYP : `Chúng tôi giấu nó`
- SRC : `Tôi là một thằng nhóc hay mắc cỡ .`
  REF : `Tôi một thằng nhóc hay mắc cỡ .`
  HYP : `Tôi một nhóc hay mắc cỡ .`
- SRC : `Tôi đợi ở đây nhé .`
  REF : `Tôi đợi ở đây .`
  HYP : `Tôi đợi ở .`
- SRC : `Bạn sẽ bị lạc đấy .`
  REF : `Bạn lạc đấy .`
  HYP : `Bạn lạc .`
- SRC : `Chúng tôi từng cãi nhau .`
  REF : `Chúng tôi cãi nhau .`
  HYP : `Chúng tôi cãi .`
- SRC : `Mùa đông đang tới .`
  REF : `Mùa đông tới .`
  HYP : `Mùa đông .`
- SRC : `Cô gái kia là Mary .`
  REF : `Cô gái kia Mary .`
  HYP : `Cô gái .`
- SRC : `Cái va li đang mở .`
  REF : `Cái va li mở .`
  HYP : `Cái va li`

## deletion_reorder  (31 mismatched)

- SRC : `1 tuần có 7 ngày .`
  REF : `1 tuần ngày 7 .`
  HYP : `1 tuần ngày 7.`
- SRC : `Trường của tôi có 3 tầng .`
  REF : `Trường tôi tầng 3 .`
  HYP : `Trường tôi tầng 3.`
- SRC : `Bạn biết lái xe hơi không ?`
  REF : `Bạn lái xe hơi biết ?`
  HYP : `Bạn lái xe hơi biết không ?`
- SRC : `Mẹ tôi 50 tuổi , đẹp và hiền .`
  REF : `Mẹ tôi tuổi 50 đẹp , hiền .`
  HYP : `Mẹ tôi tuổi 50 hiền .`
- SRC : `Hãy mở cửa sổ ra !`
  REF : `Cửa sổ mở !`
  HYP : `mở cửa sổ ra .`
- SRC : `Tôi không thua đâu !`
  REF : `Tôi thua đâu không`
  HYP : `Tôi thua .`
- SRC : `Tôi không có tội !`
  REF : `Tôi tội không`
  HYP : `Tôi có tội không`
- SRC : `Ai nói thế ?`
  REF : `nói ai ?`
  HYP : `nói Ai ?`
- SRC : `Tôi không đeo cà vạt .`
  REF : `Tôi đeo cà vạt không`
  HYP : `Tôi đeo cà .`
- SRC : `Điều đó không phải là cái mà chúng tôi đã đồng ý .`
  REF : `Điều đó phải cái mà chúng tôi đồng ý không`
  HYP : `Điều đó phải cái mà tôi đồng ý không`
- SRC : `tôi không quen với các phòng theo k = phong cách nhật của khách sạn .`
  REF : `tôi quen với phòng theo k = phong cách nhật khách sạn không`
  HYP : `tôi quen với phòng theo k phong cách nhật khách sạn không`
- SRC : `Bạn không nên .`
  REF : `Bạn nên không`
  HYP : `Bạn .`
- SRC : `Tôi không biết phải xin lỗi thế nào .`
  REF : `Tôi biết phải xin lỗi thế nào không`
  HYP : `Tôi biết phải xin lỗi không`
- SRC : `Không , cám ơn .`
  REF : `cám ơn không`
  HYP : `cám ơn Không`
- SRC : `Tôi không thể đòi hỏi gì hơn .`
  REF : `Tôi không thể đòi hỏi hơn gì`
  HYP : `Tôi không thể hỏi hơn gì`

## lexical  (9 mismatched)

- SRC : `Nhà bạn có mấy người ?`
  REF : `Gia đình bạn người mấy ?`
  HYP : `Nhà bạn người mấy ?`
- SRC : `Bạn sinh năm bao nhiêu ?`
  REF : `Bạn sinh năm mấy ?`
  HYP : `Bạn sinh năm bao nhiêu ?`
- SRC : `Người yêu của chị tôi xấu , cao và mập .`
  REF : `Chị tôi người yêu xấu , cao , mập .`
  HYP : `Người yêu chị tôi xấu cao mập .`
- SRC : `Họ sẽ giúp bọn mình`
  REF : `Họ giúp bọn mình .`
  HYP : `Họ giúp`
- SRC : `bạn không nên mặc quần len dài hôm nay .`
  REF : `bạn nên mặc quần len dài hôm nay không ?`
  HYP : `bạn nên mặc quần len dài hôm nay không`
- SRC : `Tôi sẽ đi ra ngoài vận chuyển ở Macy , vì vậy nếu ông Brown gọi , bạn sẽ vui lòng ghi lại lời nhắn cho tôi được không ?`
  REF : `cái đĩa trong tủ bát .`
  HYP : `Tôi đi ra ngoài vận chuyển ở Macy vì vậy ông Brown gọi bạn ghi lời nhắn cho tôi được không ?`
- SRC : `Chúng tôi có đồ có sẵn để khách hàng sử dụng , bao gồm máy sấy tóc và bàn ủi .`
  REF : `Chúng tôi có đồ dùng khách hàng sử dụng , bao gồm máy sấy tóc bàn ủi .`
  HYP : `Chúng tôi đồ có sẵn khách hàng sử dụng bao gồm máy sấy tóc bàn ủi .`
- SRC : `để tôi tìm vaiof lí do để rời khỏi bữa tiệc .`
  REF : `tôi tìm vài lí do rời khỏi bữa tiệc .`
  HYP : `tôi tìm vaiof lí do rời khỏi bữa tiệc .`
- SRC : `tôi muốn hạn chế các chi phí của bữa tiệc , vậy nên nó sẽ có giá hai mươi đô la cho mỗi người .`
  REF : `tôi muốn hạn chế chi phí bữa tiệc nên giá hai mươi đô la một người .`
  HYP : `tôi muốn hạn chế chi phí bữa tiệc nên nó có giá hai mươi đô la cho người .`

