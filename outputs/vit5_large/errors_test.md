# vit5_large — error examples (test)

## identical  (13 mismatched)

- SRC : `Tôi béo vãi .`
  REF : `Tôi béo vãi .`
  HYP : `Tôi béo ị .`
- SRC : `Tại sao bạn khóc ?`
  REF : `Tại sao bạn khóc ?`
  HYP : `Bạn khóc gì ?`
- SRC : `Thằng nhóc vui hẳn lên .`
  REF : `Thằng nhóc vui hẳn lên .`
  HYP : `Thằng bé vui hẳn lên .`
- SRC : `Hơn hai trăm yên một chút .`
  REF : `Hơn hai trăm yên một chút .`
  HYP : `hơn hai trăm yên một chút .`
- SRC : `Xem lại .`
  REF : `Xem lại .`
  HYP : `Xem .`
- SRC : `M-C-K-I-N-N-E-Y .`
  REF : `M-C-K-I-N-N-E-Y .`
  HYP : `M-C-K-I-N-E-Y .`
- SRC : `Luôn phải nghe cùng một kiểu trả lời khiến tôi nhàm chán .`
  REF : `Luôn phải nghe cùng một kiểu trả lời khiến tôi nhàm chán .`
  HYP : `Luôn phải nghe cùng một kiểu trả lời khiến tôi chán chán .`
- SRC : `tôi sẽ thăm nhà bạn vào ngày mai .`
  REF : `tôi sẽ thăm nhà bạn vào ngày mai .`
  HYP : `tôi thăm nhà bạn vào ngày mai .`
- SRC : `Tôi bị đau đầu .`
  REF : `Tôi bị đau đầu .`
  HYP : `Tôi đau đầu .`
- SRC : `bạn nên cẩn thận khi bạn nói chuyện với ông ấy .`
  REF : `bạn nên cẩn thận khi bạn nói chuyện với ông ấy .`
  HYP : `bạn nên cẩn thận khi bạn nói chuyện với ông .`
- SRC : `nhóm bạn bao gồm bao nhiêu thành viên ?`
  REF : `nhóm bạn bao gồm bao nhiêu thành viên ?`
  HYP : `nhóm bạn bao nhiêu thành viên ?`
- SRC : `bạn có thể sửng sốt sao ?`
  REF : `bạn có thể sửng sốt sao ?`
  HYP : `bạn có thể choáng sốt sao ?`
- SRC : `bạn Có hiểu không ?`
  REF : `bạn Có hiểu không ?`
  HYP : `bạn có hiểu không ?`

## reorder_only  (5 mismatched)

- SRC : `Con gà ăn thóc .`
  REF : `Con gà thóc ăn .`
  HYP : `Con gà ăn thóc .`
- SRC : `Ai thích nhìn thác nước ?`
  REF : `Thác nước nhìn thích ai ?`
  HYP : `Thích nhìn thác nước ai ?`
- SRC : `Tôi học 5 môn : toán , văn , lý , hóa , sinh .`
  REF : `Tôi học môn 5 : toán , văn , lý , hóa , sinh .`
  HYP : `Tôi học 5 môn toán văn lý hóa sinh .`
- SRC : `Mẹ nấu cơm ngon .`
  REF : `Mẹ cơm ngon nấu .`
  HYP : `Mẹ nấu cơm ngon .`
- SRC : `Tôi không đi chơi cùng với mẹ .`
  REF : `Tôi đi chơi cùng với mẹ không .`
  HYP : `Tôi đi chơi cùng với mẹ không`

## deletion_only  (52 mismatched)

- SRC : `Hôm qua là chủ nhật ngày 21 tháng 7 năm 2013`
  REF : `Hôm qua chủ nhật ngày 21 tháng 7 năm 2013`
  HYP : `Hôm qua chủ nhật ngày 21 tháng 7 năm 2013.`
- SRC : `Tôi nhìn thấy bố rất buồn .`
  REF : `Tôi nhìn bố buồn .`
  HYP : `Tôi nhìn thấy bố buồn .`
- SRC : `Lùi lại .`
  REF : `Lùi .`
  HYP : `L-Ê .`
- SRC : `Lại đây đi .`
  REF : `Lại đây .`
  HYP : `đây đi .`
- SRC : `Con chó của cậu đâu rồi ?`
  REF : `Con chó cậu đâu rồi ?`
  HYP : `Con chó anh đâu rồi ?`
- SRC : `Trái táo này bị hư rồi .`
  REF : `Trái táo hư rồi .`
  HYP : `quả táo hư rồi .`
- SRC : `Rồi Nam sẽ hối hận về điều này .`
  REF : `Rồi Nam hối hận về điều .`
  HYP : `rồi Nam hối hận về điều .`
- SRC : `Rồi Nam sẽ biết tay tôi .`
  REF : `Rồi Nam biết tay tôi .`
  HYP : `rồi Nam biết tay tôi .`
- SRC : `Hôm nay cậu có vẻ vui nhỉ .`
  REF : `Hôm nay cậu có vẻ vui .`
  HYP : `Hôm nay anh có vẻ vui .`
- SRC : `Bạn đến lúc nào cũng được .`
  REF : `Bạn đến lúc cũng được .`
  HYP : `Bạn đến lúc nào cũng được .`
- SRC : `Mũi của cậu đang chảy máu .`
  REF : `Mũi cậu chảy máu .`
  HYP : `Mũi anh chảy máu .`
- SRC : `Điện thoại của bạn đang reo kìa .`
  REF : `Điện thoại bạn reo .`
  HYP : `điện thoại bạn reo .`
- SRC : `tôi nên trả lại chiếc xe này ở đâu ?`
  REF : `tôi trả chiếc xe ở đâu ?`
  HYP : `tôi nên trả chiếc xe ở đâu ?`
- SRC : `tôi đã gỡ sắp ra với cái tăm bông .`
  REF : `tôi gỡ sắp ra với cái tăm bông .`
  HYP : `tôi gỡ sắp ra với cái bóng bông .`
- SRC : `anh ấy đang giơ tay lên .`
  REF : `anh giơ tay lên .`
  HYP : `anh đưa tay lên .`

## deletion_reorder  (15 mismatched)

- SRC : `1 tuần có 7 ngày .`
  REF : `1 tuần ngày 7 .`
  HYP : `1 tuần ngày 7.`
- SRC : `Trường của tôi có 3 tầng .`
  REF : `Trường tôi tầng 3 .`
  HYP : `Trường tôi tầng 3.`
- SRC : `Bạn biết lái xe hơi không ?`
  REF : `Bạn lái xe hơi biết ?`
  HYP : `Bạn lái xe biết lái xe hơi không ?`
- SRC : `Mẹ tôi 50 tuổi , đẹp và hiền .`
  REF : `Mẹ tôi tuổi 50 đẹp , hiền .`
  HYP : `Mẹ tôi tuổi 50 hiền .`
- SRC : `Hãy mở cửa sổ ra !`
  REF : `Cửa sổ mở !`
  HYP : `mở cửa sổ ra .`
- SRC : `Tôi không có tội !`
  REF : `Tôi tội không`
  HYP : `Tôi có tội không`
- SRC : `Con chó của bạn bao nhiêu tuổi ?`
  REF : `Con chó bạn tuổi bao nhiêu ?`
  HYP : `Con chó bạn tuổi mấy ?`
- SRC : `Tất cả công việc và không vui chơi biến Jacks thành một cậu bé ngớ ngẩn .`
  REF : `Tất cả công việc vui chơi biến Jacks thành một cậu bé ngớ ngẩn không`
  HYP : `Tất cả công việc vui chơi biến Jacks thành một anh bé ngớ ngẩn không`
- SRC : `tôi không quen với các phòng theo k = phong cách nhật của khách sạn .`
  REF : `tôi quen với phòng theo k = phong cách nhật khách sạn không`
  HYP : `tôi quen với phòng theo k phong cách nhật khách sạn không`
- SRC : `Chỉ làm việc mà không chơi khiến Jack trở thành một cậu nhóc đần độn .`
  REF : `làm việc mà chơi khiến Jack trở thành một cậu nhóc đần độn không`
  HYP : `làm việc mà chơi khiến Jack trở thành một anh bé đần độn không`
- SRC : `Nó sẽ làm tổn hại không thể cứu được đến sự nghiệp chính trị của anh Yano .`
  REF : `tổn hại cứu đến sự nghiệp chính trị anh Yano không .`
  HYP : `Nó làm tổn hại không thể cứu được đến sự nghiệp chính trị anh Yano .`
- SRC : `tôi đã không rõ ra lý do tại sao thư ký trả lời tôi một cách khó chịu như vậy .`
  REF : `tôi rõ ra lý do tại sao thư ký trả lời tôi một cách khó chịu như không`
  HYP : `tôi rõ ra lý do tại sao thư ký trả lời tôi một cách khó chịu như vậy không`
- SRC : `Bạn nên học về lịch sử .`
  REF : `Bạn nên lịch sử học .`
  HYP : `Bạn nên học về lịch sử .`
- SRC : `và bạn không nên bỏ lõ tòa nhà liên hợp quốc .`
  REF : `và bạn nên bỏ lõ tòa nhà liên hợp quốc không`
  HYP : `bạn nên bỏ lõ tòa nhà liên hợp quốc không`
- SRC : `tại sao bạn không để cô ấy đi ngủ ?`
  REF : `tại sao bạn cô đi ngủ không ?`
  HYP : `bạn cô đi ngủ không ?`

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
- SRC : `tại sao bạn hỏi`
  REF : `tại sao bạn hỏi .`
  HYP : `bạn hỏi tại sao bạn hỏi .`
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
  HYP : `tôi tìm vaif lí do rời khỏi bữa tiệc .`
- SRC : `tôi muốn hạn chế các chi phí của bữa tiệc , vậy nên nó sẽ có giá hai mươi đô la cho mỗi người .`
  REF : `tôi muốn hạn chế chi phí bữa tiệc nên giá hai mươi đô la một người .`
  HYP : `tôi muốn hạn chế chi phí bữa tiệc nên nó có giá hai mươi đô la cho người .`

