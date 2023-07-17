import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta-multitask')  # or 'BM-K/KoSimCSE-bert-multitask'
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask')  # or 'BM-K/KoSimCSE-bert-multitask'


def cal_score(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return float((torch.mm(a_norm, b_norm.transpose(0, 1)) * 100).item())

def get_embeddings(origin, output): #origin : 원본 텍스트 리스트, output : 생성된 텍스트 리스트
    inputs_origin = tokenizer(origin, padding=True, truncation=True, return_tensors='pt')
    inputs_output = tokenizer(output, padding=True, truncation=True, return_tensors='pt')
    embeddings_origin, _ = model(**inputs_origin, return_dict = False)
    embeddings_output, _ = model(**inputs_output, return_dict = False)
    
    return embeddings_origin, embeddings_output

