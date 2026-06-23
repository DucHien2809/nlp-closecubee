# ensemble_mbr — error examples (test)

## identical  (8 mismatched)

- SRC : `Hơn hai trăm yên một chút .`
  REF : `Hơn hai trăm yên một chút .`
  HYP : `hơn hai trăm yên một chút .`
- SRC : `Xem lại .`
  REF : `Xem lại .`
  HYP : `Xem .`
- SRC : `M-C-K-I-N-N-E-Y .`
  REF : `M-C-K-I-N-N-E-Y .`
  HYP : `M-C- K-I-N-N-N-E-Y .`
- SRC : `Chúng tôi có sáu xe buýt đi lại trong ngày .`
  REF : `Chúng tôi có sáu xe buýt đi lại trong ngày .`
  HYP : `Chúng tôi có sáu xe buýt đi trong ngày .`
- SRC : `tôi sẽ thăm nhà bạn vào ngày mai .`
  REF : `tôi sẽ thăm nhà bạn vào ngày mai .`
  HYP : `tôi thăm nhà bạn vào ngày mai .`
- SRC : `Tôi bị đau đầu .`
  REF : `Tôi bị đau đầu .`
  HYP : `Tôi đau đầu .`
- SRC : `bạn nên cẩn thận khi bạn nói chuyện với ông ấy .`
  REF : `bạn nên cẩn thận khi bạn nói chuyện với ông ấy .`
  HYP : `bạn nên cẩn thận khi bạn nói chuyện với ông .`
- SRC : `tôi không thể nói gì .`
  REF : `tôi không thể nói gì .`
  HYP : `tôi không thể nói gì`

## reorder_only  (5 mismatched)

- SRC : `Con gà ăn thóc .`
  REF : `Con gà thóc ăn .`
  HYP : `Con gà ăn thóc .`
- SRC : `Ai thích nhìn thác nước ?`
  REF : `Thác nước nhìn thích ai ?`
  HYP : `thích nhìn thác nước ai ?`
- SRC : `Tôi học 5 môn : toán , văn , lý , hóa , sinh .`
  REF : `Tôi học môn 5 : toán , văn , lý , hóa , sinh .`
  HYP : `Tôi học 5 môn : toán văn văn lý hóa sinh .`
- SRC : `Mẹ nấu cơm ngon .`
  REF : `Mẹ cơm ngon nấu .`
  HYP : `Mẹ nấu cơm ngon .`
- SRC : `Tôi không đi chơi cùng với mẹ .`
  REF : `Tôi đi chơi cùng với mẹ không .`
  HYP : `Tôi đi chơi cùng với mẹ không`

## deletion_only  (31 mismatched)

- SRC : `Hôm qua là chủ nhật ngày 21 tháng 7 năm 2013`
  REF : `Hôm qua chủ nhật ngày 21 tháng 7 năm 2013`
  HYP : `Hôm qua chủ nhật ngày 21 tháng 7 năm 2013.`
- SRC : `Tôi nhìn thấy bố rất buồn .`
  REF : `Tôi nhìn bố buồn .`
  HYP : `Tôi nhìn thấy bố buồn .`
- SRC : `Lại đây đi .`
  REF : `đây đi .`
  HYP : `Lại đây đi .`
- SRC : `Hãy nói chậm thôi .`
  REF : `nói chậm .`
  HYP : `nói chậm thôi .`
- SRC : `Lại đây đi .`
  REF : `Lại đây .`
  HYP : `Lại đây đi .`
- SRC : `Rồi Nam sẽ hối hận về điều này .`
  REF : `Rồi Nam hối hận về điều .`
  HYP : `rồi Nam hối hận về điều .`
- SRC : `Rồi Nam sẽ biết tay tôi .`
  REF : `Rồi Nam biết tay tôi .`
  HYP : `rồi Nam biết tay tôi .`
- SRC : `Bạn đến lúc nào cũng được .`
  REF : `Bạn đến lúc cũng được .`
  HYP : `Bạn đến lúc nào cũng được .`
- SRC : `tôi nên trả lại chiếc xe này ở đâu ?`
  REF : `tôi trả chiếc xe ở đâu ?`
  HYP : `tôi nên trả chiếc xe ở đâu ?`
- SRC : `thực phẩm đông lạnh thường là bao gói chân không .`
  REF : `thực phẩm đông lạnh thường bao gói chân không`
  HYP : `thực phẩm đông lạnh thường bao gói chân không .`
- SRC : `tôi cần phải chà kem lên khuôn mặt ôi .`
  REF : `tôi cần phải chà kem lên khuôn mặt .`
  HYP : `tôi cần phải chà kem lên khuôn mặt ôi .`
- SRC : `Đây là vé hành lý của ông , hãy giao cho quầy tiếp tân khi ông làm thủ tục vào khách sạn .`
  REF : `vé hành lý , giao cho quầy tiếp tân làm thủ tục vào khách sạn .`
  HYP : `vé hành lý ông giao cho quầy tiếp tân khi ông làm thủ tục vào khách sạn .`
- SRC : `ông đang làm thủ tục trả phòng hai mươi ba -hai mươi bốn .`
  REF : `ông làm thủ tục trả phòng hai mươi ba -hai mươi bốn .`
  HYP : `ông làm thủ tục trả phòng hai mươi ba hai mươi bốn .`
- SRC : `Xe buýt đi đến ga xe lửa và sân bay sử dụng đường lái xe ở phía trước lối vào phía trước .`
  REF : `Xe buýt đi đến ga xe lửa sân bay sử dụng đường lái xe ở phía trước lối vào phía trước .`
  HYP : `Xe buýt đi đến ga xe lửa sân bay sử dụng đường lái xe ở phía trước lối vào phía trước lối vào phía trước .`
- SRC : `giờ khởi hành của bạn là lúc nào ?`
  REF : `giờ khởi hành bạn lúc ?`
  HYP : `giờ khởi hành bạn lúc nào ?`

## deletion_reorder  (11 mismatched)

- SRC : `1 tuần có 7 ngày .`
  REF : `1 tuần ngày 7 .`
  HYP : `1 tuần ngày 7 ngày .`
- SRC : `Trường của tôi có 3 tầng .`
  REF : `Trường tôi tầng 3 .`
  HYP : `Trường tôi tầng 3.`
- SRC : `Bạn biết lái xe hơi không ?`
  REF : `Bạn lái xe hơi biết ?`
  HYP : `Bạn lái xe biết không ?`
- SRC : `Mẹ tôi 50 tuổi , đẹp và hiền .`
  REF : `Mẹ tôi tuổi 50 đẹp , hiền .`
  HYP : `Mẹ tôi tuổi 50 , đẹp hiền .`
- SRC : `Hãy mở cửa sổ ra !`
  REF : `Cửa sổ mở !`
  HYP : `mở cửa sổ ra .`
- SRC : `Tôi không có tội !`
  REF : `Tôi tội không`
  HYP : `Tôi có tội không`
- SRC : `Con chó của bạn bao nhiêu tuổi ?`
  REF : `Con chó bạn tuổi bao nhiêu ?`
  HYP : `Con chó bạn tuổi mấy ?`
- SRC : `tôi là brown ở phòng mười bẩy không chín .`
  REF : `tôi brown ở phòng mười bẩy chín không`
  HYP : `tôi brown ở phòng mười bẩy không chín .`
- SRC : `ngài là khách của khách sạn đúng không , thưa ngài ?`
  REF : `ngài khách khách sạn đúng ngài không ?`
  HYP : `ngài khách khách sạn đúng không ?`
- SRC : `Nó sẽ làm tổn hại không thể cứu được đến sự nghiệp chính trị của anh Yano .`
  REF : `tổn hại cứu đến sự nghiệp chính trị anh Yano không .`
  HYP : `Nó làm tổn hại không thể cứu được đến sự nghiệp chính trị anh Yano .`
- SRC : `Bạn nên học về lịch sử .`
  REF : `Bạn nên lịch sử học .`
  HYP : `Bạn nên học về lịch sử .`

## lexical  (8 mismatched)

- SRC : `Nhà bạn có mấy người ?`
  REF : `Gia đình bạn người mấy ?`
  HYP : `Nhà bạn người mấy ?`
- SRC : `Bạn sinh năm bao nhiêu ?`
  REF : `Bạn sinh năm mấy ?`
  HYP : `Bạn sinh năm bao nhiêu ?`
- SRC : `Người yêu của chị tôi xấu , cao và mập .`
  REF : `Chị tôi người yêu xấu , cao , mập .`
  HYP : `Người yêu chị tôi xấu cao mập .`
- SRC : `bạn không nên mặc quần len dài hôm nay .`
  REF : `bạn nên mặc quần len dài hôm nay không ?`
  HYP : `bạn nên mặc quần len dài hôm nay không`
- SRC : `Tôi sẽ đi ra ngoài vận chuyển ở Macy , vì vậy nếu ông Brown gọi , bạn sẽ vui lòng ghi lại lời nhắn cho tôi được không ?`
  REF : `cái đĩa trong tủ bát .`
  HYP : `Tôi đi ra ngoài vận chuyển ở Macy vì vậy ông Brown gọi bạn ghi lời nhắn cho tôi được không ?`
- SRC : `Chúng tôi có đồ có sẵn để khách hàng sử dụng , bao gồm máy sấy tóc và bàn ủi .`
  REF : `Chúng tôi có đồ dùng khách hàng sử dụng , bao gồm máy sấy tóc bàn ủi .`
  HYP : `Chúng tôi có đồ có sẵn khách hàng sử dụng bao gồm máy sấy tóc bàn ủi .`
- SRC : `để tôi tìm vaiof lí do để rời khỏi bữa tiệc .`
  REF : `tôi tìm vài lí do rời khỏi bữa tiệc .`
  HYP : `tôi tìm vaiof lí do rời khỏi bữa tiệc .`
- SRC : `tôi muốn hạn chế các chi phí của bữa tiệc , vậy nên nó sẽ có giá hai mươi đô la cho mỗi người .`
  REF : `tôi muốn hạn chế chi phí bữa tiệc nên giá hai mươi đô la một người .`
  HYP : `tôi muốn hạn chế chi phí bữa tiệc nên nó có giá hai mươi đô la cho người .`

