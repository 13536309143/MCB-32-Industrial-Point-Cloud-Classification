import numpy as np


def classification_metrics(labels, probs, classes):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    predicted = probs.argmax(1)
    k = len(classes)
    cm = np.bincount(labels*k+predicted, minlength=k*k).reshape(k, k)
    support, predicted_count = cm.sum(1), cm.sum(0)
    tp = cm.diagonal()
    recall = np.divide(tp, support, out=np.zeros(k, dtype=float), where=support > 0)
    precision = np.divide(tp, predicted_count, out=np.zeros(k, dtype=float), where=predicted_count > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(k, dtype=float), where=precision+recall > 0)
    confidence = probs.max(1)
    correct = predicted == labels
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    onehot = np.eye(k)[labels]
    return dict(samples=len(labels), oa=float(correct.mean()), macc=float(recall.mean()),
                macro_f1=float(f1.mean()), weighted_f1=float(np.dot(f1, support)/len(labels)),
                macro_precision=float(precision.mean()), nll=float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
                top3_oa=float(np.any(np.argsort(probs, axis=1)[:, -min(3, k):] == labels[:, None], axis=1).mean()),
                brier=float(((probs-onehot)**2).sum(1).mean()), ece_15bins=float(ece),
                confusion_matrix=cm.tolist(),
                per_class=[dict(index=i, name=classes[i], support=int(support[i]),
                                predicted_count=int(predicted_count[i]), precision=float(precision[i]),
                                recall=float(recall[i]), f1=float(f1[i])) for i in range(k)])
