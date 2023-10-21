import torch
from transformers import AutoModel, AutoTokenizer
from flask_cors import CORS
from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
CORS(app)
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


@app.route("/api/query")
def answer():
    prompt = str(request.args.get("prompt"))
    print(prompt)
    response, history = model.chat(
        tokenizer,
        prompt,
        history=[],
        max_length=4096,
        top_p=0.7,
        temperature=0.5,
    )
    torch_gc()
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(port=8189)
