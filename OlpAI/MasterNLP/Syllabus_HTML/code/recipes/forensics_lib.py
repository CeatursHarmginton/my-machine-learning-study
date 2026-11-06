from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from transformers import AutoTokenizer

def build_subword_forensic_clf(train_texts: list[str], vocab_size: int = 30000):
    """Chiến thuật Top-1 Detect AI Text dùng 100% thư viện OlpAI'25 (transformers + sklearn)."""
    sub_tok = AutoTokenizer.from_pretrained("gpt2").train_new_from_iterator(train_texts, vocab_size=vocab_size)
    tok_docs = [" ".join(map(str, sub_tok.encode(t))) for t in train_texts]
    vec = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, token_pattern=r"\S+")
    clf = VotingClassifier(
        [("nb", MultinomialNB(alpha=0.02)), ("sgd", SGDClassifier(loss="modified_huber", random_state=42))],
        weights=[0.35, 0.65], voting="soft",
    )
    return sub_tok, vec, vec.fit_transform(tok_docs), clf
