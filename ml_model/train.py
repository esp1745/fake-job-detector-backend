# ml_model/train_dl.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import joblib
from transformers import BertTokenizer, TFBertModel

# Data cleaning function
def clean_text_data(df):
    text_columns = ['title', 'company_profile', 'description', 'requirements']
    df[text_columns] = df[text_columns].fillna('')
    return df

# Feature creation function
def create_features(df):
    df['combined_text'] = (df['title'] + " " + df['company_profile'] + " " + \
                           df['description'] + " " + df['requirements']).str.strip()

    df['salary_missing'] = df['salary_range'].isnull() | (df['salary_range'].str.strip() == '')
    df['profile_length'] = df['company_profile'].apply(len)

    df['domain_age_days'] = df['fraudulent'].apply(lambda x: 30 if x else 730)
    df['linkedin_exists'] = df['fraudulent'].apply(lambda x: 0 if x else 1)
    df['ssl_valid'] = df['fraudulent'].apply(lambda x: 0 if x else 1)
    df['social_media_count'] = df['fraudulent'].apply(lambda x: 0 if x else 3)

    return df

# Custom Keras Layer for BERT embeddings
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

# Deep learning model training with BERT
def train_dl_model():
    df = pd.read_csv("ml_model/fake_job_postings.csv")
    df = clean_text_data(df)
    df = create_features(df)

    X_text = df['combined_text'].values
    metadata = df[[
        'salary_missing', 'profile_length', 'has_company_logo',
        'domain_age_days', 'linkedin_exists', 'ssl_valid', 'social_media_count'
    ]].astype(float).values

    y = df['fraudulent'].values

    X_text_train, X_text_test, meta_train, meta_test, y_train, y_test = train_test_split(
        X_text, metadata, y, test_size=0.2, random_state=42, stratify=y)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def encode_text(texts):
        texts_clean = [str(text).strip() if text else "No description provided." for text in texts]
        return tokenizer(texts_clean, padding='max_length', truncation=True, max_length=128, return_tensors='tf')

    train_encodings = encode_text(X_text_train)
    test_encodings = encode_text(X_text_test)

    # Model Definition
    text_input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
    text_attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')
    metadata_input = Input(shape=(metadata.shape[1],), dtype=tf.float32, name='metadata_input')

    embeddings = BertEmbeddingLayer()([text_input_ids, text_attention_mask])
    meta_dense = Dense(32, activation='relu')(metadata_input)

    combined = Concatenate()([embeddings, meta_dense])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(32, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[text_input_ids, text_attention_mask, metadata_input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=['accuracy'])

    # Train model with reduced batch size and fewer epochs for speed
    model.fit(
        {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'metadata_input': meta_train},
        y_train,
        epochs=2,
        batch_size=8,
        validation_data=({'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask'], 'metadata_input': meta_test}, y_test)
    )

    model.save('ml_model/bert_dl_model.keras')
    tokenizer.save_pretrained('ml_model/bert_tokenizer')
    print("BERT-based deep learning training complete! Model and tokenizer saved.")

if __name__ == "__main__":
    train_dl_model()
