import torch
from sklearn.metrics import accuracy_score, f1_score

# compute_metrics 함수: Trainer 에서 metric 을 계산하기 위해 쓰이는 함수
def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)

    # calculate f1 score using sklearn's function
    f1 = f1_score(labels, preds, average="micro")

    return {
        "accuracy": acc,
        "f1": f1,
    }

