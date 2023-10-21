from transformers import AutoTokenizer, AutoModel
import json
import torch
from tqdm import tqdm
from datetime import datetime


def generate_articles(dev_data, batch_size, tokenizer, model, max_length=4096, skip_special_tokens=True, clean_up_tokenization_spaces=True):
    res = []
    start = datetime.now()
    for i in range(0, len(dev_data), batch_size):
        if i % 1 == 0:
            print(i, '/', len(dev_data), datetime.now() - start)
        batch = dev_data[i:i+batch_size]
        batch_text = []
        for item in batch:
            input_text = "基本案情：\n" + item['fact'] + "\n请根据基本案情列举出相关的法条，法条数目不超过5条，法条格式为：《法律名称》+第几条+【罪名】+罪名描述。"
            batch_text.append(tokenizer.bos_token + input_text if tokenizer.bos_token!=None else input_text)
        max_length = min(max_length, max([len(i) for i in batch_text]))
        print(max_length)
        features = tokenizer(batch_text, padding=True, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = features['input_ids'].to("cuda")
        # attention_mask = features['attention_mask'].to("cuda")
        print(222)
        output_texts = model.generate(
            input_ids=input_ids,
            min_new_tokens=1,
            max_new_tokens=3000,
            early_stopping=True,
            temperature=0.0
        )
        print(333)
        output_texts = tokenizer.batch_decode(
            output_texts.cpu().numpy().tolist(),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        print(444)
        for i in range(len(output_texts)):
            input_text = batch_text[i]
            input_text = input_text.replace(tokenizer.bos_token, "")
            predict_text = output_texts[i][len(input_text):]
            res.append({"input": input_text, "predict": predict_text})
        print(res[-1]["predict"])
    return res


if __name__ == '__main__':
    original_path = 'dataset/train-1000.json'
    with open(original_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(
        "/data03/dengwentao-slurm/Legal_LLM/Library/model/ChatGLM", trust_remote_code=True)
    model = (
        AutoModel.from_pretrained("/data03/dengwentao-slurm/Legal_LLM/Library/model/ChatGLM",
                                  trust_remote_code=True).half().cuda()
    )
    model.eval()
    with torch.no_grad():
        res = generate_articles(dev_data, batch_size=2, tokenizer=tokenizer, model=model)
    for i in res[:10]:
        print(i)
    print(len(res))

