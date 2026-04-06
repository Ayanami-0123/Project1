from datasets import load_dataset
from transformers import AutoTokenizer

parquet_path = "/data/home/huangqiyuan/data/math_train_fixed.parquet"
model_path = "/data/home/huangqiyuan/models/qwen/Qwen2.5-7B-Instruct"  # 与训练配置一致

ds = load_dataset("parquet", data_files=parquet_path)["train"]
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

messages = ds[0]["prompt"]
# 若块 A 发现是 str，这里先: messages = json.loads(messages)

print("=== B1: apply_chat_template 全文（add_generation_prompt=True）===")
text = tok.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)
print(text[:2000])
print("... total chars:", len(text))

print("=== B2: 是否含你的 system 关键词（自己改关键字）===")
print("intellegent" in text or "math" in text.lower())  # 按你的 system 改

print("=== B3: encode 长度（与 max_prompt_length 对比）===")
ids = tok.encode(text, add_special_tokens=False)
print("len(ids) =", len(ids))