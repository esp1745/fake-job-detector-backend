import os
import json
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
import faiss

# ---- Custom Embedding Layer ----
class BertEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BertEmbeddingLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    def get_config(self):
        config = super().get_config()
        return config

# ---- Load Model and Tokenizer ----
model_path = os.path.join(os.path.dirname(__file__), 'bert_dl_model.keras')
tokenizer_path = os.path.join(os.path.dirname(__file__), 'bert_tokenizer')
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = load_model(model_path, custom_objects={'BertEmbeddingLayer': BertEmbeddingLayer})

# ---- RAG Setup: Embedding model + FAISS ----
rag_model = SentenceTransformer("all-MiniLM-L6-v2")
knowledge_base = [
    "Scam job posts often avoid providing salary details.",
    "Legitimate companies usually have a LinkedIn presence.",
    "Short or missing company profiles are red flags in many scams.",
    "Domains that are recently registered can indicate fake companies.",
    "Job scams often pressure applicants to act urgently or pay upfront."
]
knowledge_embeddings = rag_model.encode(knowledge_base)
index = faiss.IndexFlatL2(knowledge_embeddings.shape[1])
index.add(np.array(knowledge_embeddings))

# ---- Simple RAG-based Generator ----
def retrieve_rag_context(text_query, top_k=2):
    query_vec = rag_model.encode([text_query])
    _, indices = index.search(query_vec, top_k)
    return [knowledge_base[i] for i in indices[0]]

def generate_explanation_from_rag(metadata, retrieved):
    explanation = "Based on the retrieved knowledge and job features, this post appears suspicious due to: "
    reasons = []
    if metadata['salary_missing']:
        reasons.append("missing salary info")
    if metadata['profile_length'] < 50:
        reasons.append("short company profile")
    if not metadata['linkedin_exists']:
        reasons.append("no LinkedIn presence")
    if metadata['domain_age_days'] < 365:
        reasons.append("newly registered domain")
    if not metadata['ssl_valid']:
        reasons.append("no valid SSL certificate")
    if metadata['social_media_count'] < 1:
        reasons.append("low social media presence")

    explanation += ", ".join(reasons) + ".\n\nRetrieved Context:\n"
    for line in retrieved:
        explanation += f"- {line}\n"
    return explanation.strip()

# ---- Prediction Function ----
def predict_job(text: str, metadata: dict):
    text = str(text or "").strip() or "No description provided."

    # Encode text with BERT tokenizer
    encoding = tokenizer(
        [text],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='tf'
    )

    # Prepare metadata
    metadata_features = np.array([[
        int(metadata['salary_missing']),
        metadata['profile_length'],
        int(metadata['has_company_logo']),
        metadata['domain_age_days'],
        int(metadata['linkedin_exists']),
        int(metadata['ssl_valid']),
        metadata['social_media_count']
    ]], dtype=float)

    # Predict with model
    prediction_prob = model.predict({
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'metadata_input': metadata_features
    })[0][0]
    prediction = int(prediction_prob >= 0.5)

    # RAG Explanation
    rag_query = f"Job Text: {text}\nMetadata: {json.dumps(metadata)}"
    retrieved_docs = retrieve_rag_context(rag_query)
    explanation = generate_explanation_from_rag(metadata, retrieved_docs)

    return {
        "prediction": prediction,
        "probability": float(prediction_prob),
        "explanation": explanation,
        "evidence": retrieved_docs
    }
