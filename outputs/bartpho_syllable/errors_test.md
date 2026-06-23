# bartpho_syllable — error examples (test)

## identical  (11 mismatched)

- SRC : `Tôi có hai con mèo .`
  REF : `Tôi có hai con mèo .`
  HYP : `Tôi mèo hai .`
- SRC : `Hơn hai trăm yên một chút .`
  REF : `Hơn hai trăm yên một chút .`
  HYP : `hai trăm yên một chút .`
- SRC : `Xem lại .`
  REF : `Xem lại .`
  HYP : `Xem .`
- SRC : `vợ bạn khỏe không ?`
  REF : `vợ bạn khỏe không ?`
  HYP : `vợ bạn không ?`
- SRC : `Chúng tôi có sáu xe buýt đi lại trong ngày .`
  REF : `Chúng tôi có sáu xe buýt đi lại trong ngày .`
  HYP : `Chúng tôi xe buýt đi trong ngày .`
- SRC : `tôi sẽ thăm nhà bạn vào ngày mai .`
  REF : `tôi sẽ thăm nhà bạn vào ngày mai .`
  HYP : `tôi thăm nhà bạn vào ngày mai .`
- SRC : `tôi phải trả bao nhiêu khi tôi hủy bỏ việc đặt trước ở khách sạn ?`
  REF : `tôi phải trả bao nhiêu khi tôi hủy bỏ việc đặt trước ở khách sạn ?`
  HYP : `tôi phải trả bao nhiêu khi tôi bỏ việc đặt trước ở khách sạn ?`
- SRC : `Tôi bị đau đầu .`
  REF : `Tôi bị đau đầu .`
  HYP : `Tôi đau đầu .`
- SRC : `bạn nên cẩn thận khi bạn nói chuyện với ông ấy .`
  REF : `bạn nên cẩn thận khi bạn nói chuyện với ông ấy .`
  HYP : `bạn nên cẩn thận khi bạn nói chuyện với ông .`
- SRC : `tôi không thể nói gì .`
  REF : `tôi không thể nói gì .`
  HYP : `tôi không thể nói gì`
- SRC : `tôi có thể cược vé cho buổi hòa nhạc tối nay không ?`
  REF : `tôi có thể cược vé cho buổi hòa nhạc tối nay không ?`
  HYP : `tôi có thể cược vé cho buổi nhạc tối nay không ?`

## reorder_only  (6 mismatched)

- SRC : `Con gà ăn thóc .`
  REF : `Con gà thóc ăn .`
  HYP : `Con gà ăn thóc .`
- SRC : `Ai thích nhìn thác nước ?`
  REF : `Thác nước nhìn thích ai ?`
  HYP : `thích nhìn thác nước ai ?`
- SRC : `Tôi thích nhất màu trắng .`
  REF : `Tôi màu trắng thích nhất .`
  HYP : `Tôi thích nhất màu trắng .`
- SRC : `Tôi học 5 môn : toán , văn , lý , hóa , sinh .`
  REF : `Tôi học môn 5 : toán , văn , lý , hóa , sinh .`
  HYP : `Tôi học 5 môn : toán văn văn , lý , hóa sinh .`
- SRC : `Mẹ nấu cơm ngon .`
  REF : `Mẹ cơm ngon nấu .`
  HYP : `Mẹ nấu cơm ngon .`
- SRC : `Tôi không đi chơi cùng với mẹ .`
  REF : `Tôi đi chơi cùng với mẹ không .`
  HYP : `Tôi đi chơi cùng với mẹ không`

## deletion_only  (45 mismatched)

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
- SRC : `Nam đã đạt được một thỏa thuận .`
  REF : `Nam đạt được một thỏa thuận .`
  HYP : `Nam đạt được một thuận .`
- SRC : `Rồi Nam sẽ hối hận về điều này .`
  REF : `Rồi Nam hối hận về điều .`
  HYP : `rồi Nam hối hận về điều .`
- SRC : `Rồi Nam sẽ biết tay tôi .`
  REF : `Rồi Nam biết tay tôi .`
  HYP : `Nam biết tay tôi .`
- SRC : `Khi nào thì Nam cần nó ?`
  REF : `Khi Nam nó ?`
  HYP : `Khi Nam cần nó ?`
- SRC : `tôi nên trả lại chiếc xe này ở đâu ?`
  REF : `tôi trả chiếc xe ở đâu ?`
  HYP : `tôi nên trả chiếc xe ở đâu ?`
- SRC : `đây là chìa khóa phòng của bạn .`
  REF : `chìa khóa phòng bạn .`
  HYP : `chìa phòng bạn .`
- SRC : `thực phẩm đông lạnh thường là bao gói chân không .`
  REF : `thực phẩm đông lạnh thường bao gói chân không`
  HYP : `thực phẩm đông lạnh thường bao gói chân không .`
- SRC : `Nó bị hủy về lý do gì ?`
  REF : `Nó hủy về lý do gì ?`
  HYP : `Nó về lý do gì ?`
- SRC : `bạn đan bị khóa à ?`
  REF : `bạn đan khóa ?`
  HYP : `bạn đan ?`
- SRC : `tôi cần phải chà kem lên khuôn mặt ôi .`
  REF : `tôi cần phải chà kem lên khuôn mặt .`
  HYP : `tôi cần phải chà kem lên khuôn mặt ôi .`

## deletion_reorder  (17 mismatched)

- SRC : `1 tuần có 7 ngày .`
  REF : `1 tuần ngày 7 .`
  HYP : `1 tuần ngày 7.`
- SRC : `Trường của tôi có 3 tầng .`
  REF : `Trường tôi tầng 3 .`
  HYP : `Trường tôi tầng 3.`
- SRC : `Bạn biết lái xe hơi không ?`
  REF : `Bạn lái xe hơi biết ?`
  HYP : `Bạn lái xe biết không ?`
- SRC : `Mẹ tôi 50 tuổi , đẹp và hiền .`
  REF : `Mẹ tôi tuổi 50 đẹp , hiền .`
  HYP : `Mẹ tôi tuổi 50 , đẹp , hiền .`
- SRC : `Hãy mở cửa sổ ra !`
  REF : `Cửa sổ mở !`
  HYP : `mở cửa sổ ra .`
- SRC : `Tôi không có tội !`
  REF : `Tôi tội không`
  HYP : `Tôi có tội không`
- SRC : `Con chó của bạn bao nhiêu tuổi ?`
  REF : `Con chó bạn tuổi bao nhiêu ?`
  HYP : `Con chó bạn tuổi mấy ?`
- SRC : `nếu tôi trúng số , tôi sẽ không bao giờ phải làm việc nữa .`
  REF : `tôi trúng số tôi bao giờ phải làm việc nữa không`
  HYP : `tôi trúng số tôi phải làm việc nữa không`
- SRC : `bạn sẽ nhận được bao nhiêu tiền nếu bạn thắng ?`
  REF : `bạn nhận được tiền bao nhiêu bạn thắng ?`
  HYP : `bạn nhận được bao nhiêu bạn thắng ?`
- SRC : `tôi là brown ở phòng mười bẩy không chín .`
  REF : `tôi brown ở phòng mười bẩy chín không`
  HYP : `tôi brown ở phòng mười bẩy không chín .`
- SRC : `Nếu không , chỉ cần cho tôi bất cứ phòng nào có sẵn là được .`
  REF : `cho tôi bất cứ phòng có sẵn được không`
  HYP : `cho tôi bất cứ phòng có sẵn không`
- SRC : `ngài là khách của khách sạn đúng không , thưa ngài ?`
  REF : `ngài khách khách sạn đúng ngài không ?`
  HYP : `ngài khách khách sạn đúng không ?`
- SRC : `Nó sẽ làm tổn hại không thể cứu được đến sự nghiệp chính trị của anh Yano .`
  REF : `tổn hại cứu đến sự nghiệp chính trị anh Yano không .`
  HYP : `Nó làm tổn hại không thể cứu được đến sự nghiệp chính trị anh Yano .`
- SRC : `nhà điều hành , đường truyền này không rõ lắm .`
  REF : `nhà điều hành đường truyền rõ không`
  HYP : `nhà điều hành đường truyền không rõ .`
- SRC : `tôi không thích câu trả lời mơ hồ như vậy .`
  REF : `tôi thích câu trả lời mơ hồ như vậy không`
  HYP : `tôi thích câu trả lời mơ hồ như không`

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

