import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from flask_cors import CORS
from flask import Flask, jsonify, request
from data_utils import *

app = Flask(__name__)
CORS(app)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class info_retrieval:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Lawformer")
        self.model = AutoModel.from_pretrained(
            "Lawformer")
        self.model.to(DEVICE)
        self.model.eval()
        self.calc = torch.nn.CosineSimilarity(dim=0)
        self.data = []
        # print([i for i in self.df[listname]])
        self.articles_dataset = Dataset(self.df)
        print(self.articles_dataset[0])
        self.articles_data = DataLoader(self.articles_dataset, batch_size=32)
        result_data = []
        with torch.no_grad():
            for data in self.articles_data:
                print(data['attention_mask'].squeeze().shape)
                output = self.model(input_ids=data['input_ids'].squeeze().to(
                    DEVICE), attention_mask=data['attention_mask'].squeeze().to(DEVICE))
                print(output.last_hidden_state[:, 0, :].shape)
                result_data.append(output.last_hidden_state[:, 0, :])
        self.data = torch.cat(result_data)
        print(self.data.shape)

    def query(self, query, kth):
        query = query[:3000]
        law_list, tensor_list, results, answer_list = [], [], [], []
        query_vec = self.tokenizer(query, return_tensors="pt")
        query_vec = self.model(input_ids=query_vec['input_ids'].to(
            DEVICE), attention_mask=query_vec['attention_mask'].to(DEVICE)).last_hidden_state[:, 0, :]
        if len(law_list) == 0:
            for context in self.data:
                results.append(self.calc(context, query_vec.squeeze()))
            results_rank = sorted(range(len(results)), key=results.__getitem__)
            results_rank = results_rank[::-1]
            for i in results_rank[:kth]:
                answer_list.append(
                    {
                        "title": self.df["title"][i],
                        "accu": self.df["accu"][i],
                        "context": self.df["context"][i],
                        "all": self.df["all"][i],
                    }
                )
        else:
            kth = min(kth, len(law_list))
            for context in tensor_list:
                results.append(self.calc(context, query_vec.squeeze()))
            results_rank = sorted(range(len(results)), key=results.__getitem__)
            results_rank = results_rank[::-1]
            for i in results_rank[:kth]:
                answer_list.append(
                    {
                        "title": law_list[i]["title"],
                        "accu": law_list[i]["accu"],
                        "context": law_list[i]["context"],
                        "all": law_list[i]["all"],
                    }
                )

        return answer_list


ir = info_retrieval(filename="output.csv")

@app.route('/api/query')
def answer():
    query = str(request.args.get('query'))
    kth = int(str(request.args.get('kth')))
    answer = ir.query(query, kth)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
