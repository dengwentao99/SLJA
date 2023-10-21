import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = (
    AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
)
model.eval()


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def answer(prompt):
    # prompt = str(request.args.get("prompt"))
    response, history = model.chat(
        tokenizer,
        prompt,
        history=[],
        max_length=4096,
    )
    return response


if __name__ == "__main__":
    print(answer("你是谁?"))
