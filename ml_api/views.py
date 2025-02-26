import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import tensorflow as tf
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from django.http import JsonResponse
from transformers import DistilBertTokenizer, TFDistilBertModel, AutoTokenizer, AutoModelForSequenceClassification, RobertaTokenizer, TFRobertaModel


# Load BiLSTM model
bilstm_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
distilbert.trainable = False  # Freeze DistilBERT layers
bilstm_model_path = os.path.join(os.path.dirname(__file__), 'C:/Users/Admin/dev/trail/ml_api/bilstm_model.h5')
bilstm_model = tf.keras.models.load_model(bilstm_model_path)

# Load RL model
base_dir = 'C:/Users/Admin/dev/trail/ml_api/trained_model2'
model_path = os.path.join(base_dir, 'distilroberta_model.pth')
tokenizer_dir = os.path.join(base_dir, 'distilroberta_tokenizer')
tokenizer_rl = AutoTokenizer.from_pretrained(tokenizer_dir)
model_rl = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=2)
model_rl.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model_rl.eval()

# Load PU model
pu_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = TFRobertaModel.from_pretrained('roberta-base')
roberta_model.trainable = False
pu_classifier = xgb.Booster()
pu_classifier.load_model(os.path.join(os.path.dirname(__file__), 'C:/Users/Admin/dev/trail/ml_api/xgboost_spam_filter.model'))

def classify_with_bilstm(text):
    inputs = bilstm_tokenizer(text, return_tensors="tf", padding="max_length", truncation=True, max_length=128)
    embeddings = distilbert(inputs).last_hidden_state
    prediction = bilstm_model.predict(embeddings)
    spam_prob = float(tf.nn.sigmoid(prediction)[0][0])
    return max(0, min(1, spam_prob))

def classify_with_rl(text):
    inputs = tokenizer_rl(text, return_tensors="pt")
    outputs = model_rl(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return max(0, min(1, float(probs[0][1])))

def classify_with_pu(text):
    inputs = pu_tokenizer(text, return_tensors="tf", truncation=True, max_length=128)
    embeddings = roberta_model(inputs).last_hidden_state[:, 0, :]
    dmatrix = xgb.DMatrix(embeddings.numpy())
    pu_probs = pu_classifier.predict(dmatrix)
    return max(0, min(1, float(pu_probs[0]) if pu_probs.size > 0 else 0.5))

def soft_voting_ensemble(text):
    probs_bilstm = classify_with_bilstm(text)
    probs_rl = classify_with_rl(text)
    probs_pu = classify_with_pu(text)
    return (probs_bilstm + probs_rl + probs_pu) / 3



def generate_bar_chart(metrics, title):
    fig, ax = plt.subplots()
    ax.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red'])
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def bulk_classify(request):
    if request.method == "POST":
        try:
            file = request.FILES.get("file")
            if not file:
                return JsonResponse({'error': 'No file uploaded'}, status=400)

            df = pd.read_csv(file).dropna()
            if "sms" not in df.columns or "label" not in df.columns:
                return JsonResponse({'error': 'CSV must contain "sms" and "label" columns'}, status=400)

            smses = df["sms"].astype(str).tolist()
            true_labels = df["label"].astype(int).tolist()

            model_results = {}
            ensemble_probs = []

            for model_name, classify_func in zip(["BiLSTM", "Reinforcement Learning", "PU Learning"],
                                                 [classify_with_bilstm, classify_with_rl, classify_with_pu]):
                probs = [classify_func(sms) for sms in smses]
                print(f"{model_name} Probabilities: {probs}")  # Debugging output
                model_metrics = {
                    "accuracy": accuracy_score(true_labels, [1 if p >= 0.5 else 0 for p in probs]) * 100,
                    "precision": precision_score(true_labels, [1 if p >= 0.5 else 0 for p in probs], zero_division=0) * 100,
                    "recall": recall_score(true_labels, [1 if p >= 0.5 else 0 for p in probs], zero_division=0) * 100,
                    "f1_score": f1_score(true_labels, [1 if p >= 0.5 else 0 for p in probs], zero_division=0) * 100
                }
                model_results[model_name] = {
                    "description": f"{model_name} model for spam detection.",
                    "table": [{"sms": sms, "probability": round(prob * 100, 2)} for sms, prob in zip(smses, probs)],
                    "metrics": model_metrics,
                    "chart": generate_bar_chart(model_metrics, f"Metrics for {model_name}")
                }
                ensemble_probs.append(probs)

            ensemble_final_probs = np.mean(np.array(ensemble_probs), axis=0)
            print(f"Ensemble Probabilities: {ensemble_final_probs}")  # Debugging output
            ensemble_metrics = {
                "accuracy": accuracy_score(true_labels, [1 if p >= 0.5 else 0 for p in ensemble_final_probs]) * 100,
                "precision": precision_score(true_labels, [1 if p >= 0.5 else 0 for p in ensemble_final_probs], zero_division=0) * 100,
                "recall": recall_score(true_labels, [1 if p >= 0.5 else 0 for p in ensemble_final_probs], zero_division=0) * 100,
                "f1_score": f1_score(true_labels, [1 if p >= 0.5 else 0 for p in ensemble_final_probs], zero_division=0) * 100
            }
            ensemble_results = [{"sms": sms, "ensemble": "SPAM" if prob >= 0.5 else "NOT SPAM"} for sms, prob in zip(smses, ensemble_final_probs)]

            return JsonResponse({"models": model_results, "ensemble": {"table": ensemble_results, "metrics": ensemble_metrics, "chart": generate_bar_chart(ensemble_metrics, f"Metrics for {"Ensemble Model"}")  }})
        except Exception as e:
            print(f"Bulk Classification Error: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def classify_text(request):
    if request.method == "POST":
        try:
            # Parse the JSON request body
            data = json.loads(request.body)
            text = data.get('text', '')
            model_selected = data.get('modelSelect', 'bilstm')  # Default to BiLSTM

            if not text:
                return JsonResponse({'error': 'No text provided'}, status=400)

            # Get predictions based on the selected model
            if model_selected == 'bilstm':
                probability = classify_with_bilstm(text)
            elif model_selected == 'rl':
                probability = classify_with_rl(text)
            elif model_selected == 'pu':
                probability = classify_with_pu(text)
            elif model_selected == 'ensemble':
                probability = soft_voting_ensemble(text)
            else:
                return JsonResponse({'error': 'Invalid model selected'}, status=400)

            # Return the prediction and probability as a JSON response
            return JsonResponse({
                'prediction': "SPAM" if probability >= 0.5 else "NOT SPAM",
                'probability': probability
            })
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    # Return an error for non-POST requests
    return JsonResponse({'error': 'Invalid request method'}, status=405)