import torch
import torch.nn.functional as F
from gensim.corpora import Dictionary
from gensim.models import OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
from transformers import AutoModel, AutoTokenizer

# 1. Chỉ mục từ khóa Okapi BM25 có sẵn trong thư viện chuẩn gensim của OlpAI'25:
def build_gensim_bm25(corpus_tokens: list[list[str]]):
    dct = Dictionary(corpus_tokens)
    bm25 = OkapiBM25Model(dictionary=dct, k1=1.5, b=0.75)
    index = SparseMatrixSimilarity(bm25[list(map(dct.doc2bow, corpus_tokens))], num_docs=len(corpus_tokens), num_terms=len(dct))
    return dct, bm25, index

# 2. Bi-Encoder trích xuất vector câu chuẩn hóa L2 chỉ bằng transformers + torch (không cần sentence-transformers):
@torch.no_grad()
def encode_texts(texts: list[str], tok, model, max_len: int = 256) -> torch.Tensor:
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(model.device)
    h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).type_as(h)
    return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-6), p=2, dim=-1)
