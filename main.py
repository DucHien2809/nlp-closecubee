from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ====== 1. Load model and tokenizer ======
# Sửa lại đường dẫn tới thư mục model BARTpho bạn vừa train xong
# Nếu chạy trên Kaggle thì dùng: "/kaggle/working/bartpho_vsl_best_model"
model_dir = "./vit5_sign_grammar_finetuned_10_epochs" 

# Với BARTpho không cần use_fast=False
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# ====== 2. Move model to GPU ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ====== 3. Tạo hàm chuyển đổi để dễ tái sử dụng ======
def translate_to_vsl(sentence):
    # Với BART, không cần thêm tiền tố (prefix) hay hậu tố (</s>) thủ công
    text = sentence.strip()
    
    # Tokenize đầu vào
    encoding = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=128,       # Độ dài 128 là đủ cho câu giao tiếp
        truncation=True
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # ====== 4. Generate output ======
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=5,             # Dùng Beam Search = 5 để dịch ngữ pháp chuẩn xác nhất
        early_stopping=True,     # Dừng sinh từ khi gặp token kết thúc câu
        no_repeat_ngram_size=2   # Tránh việc model bị lặp từ (VD: "yêu yêu")
    )

    # ====== 5. Decode output ======
    # Chỉ lấy kết quả tốt nhất (chỉ số 0)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return decoded

# ====== 6. Chạy thử nghiệm các ví dụ ======
test_sentences = [
    "Tôi rất yêu động vật",     # Kỳ vọng: Tôi động vật yêu
    "Tôi 19 tuổi",              # Kỳ vọng: Tôi tuổi 19
    "Mẹ đang nấu cơm ở trong bếp",
    "Ngày mai tôi sẽ đi học",
    "Tôi là Trần Khánh Duy"
]

print("="*50)
for sentence in test_sentences:
    vsl_output = translate_to_vsl(sentence)
    print(f"Câu gốc nói/viết : {sentence}")
    print(f"Ngữ pháp VSL     : {vsl_output}")
    print("-" * 50)