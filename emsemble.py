import joblib
from scipy.stats import mode
import numpy as np
import torch
from torch_NN import CNN
from collections import defaultdict

log_reg_model = joblib.load('mnist_log_reg_model.joblib')
svm_model = joblib.load('mnist_svm_model.joblib')
rf_model = joblib.load('mnist_rf_model.joblib')
knn_model = joblib.load('mnist_knn_model.joblib')
naive_bayes_model = joblib.load('mnist_naive_bayes_model.joblib')

device = torch.device("cpu")
torch_model = CNN().to(device)
torch_model.load_state_dict(torch.load('model_epoch_40.pth', map_location=torch.device('cpu')))
torch_model.eval()

weights = {
    'log_reg': 1.0,  # 逻辑回归模型的权重
    'svm': 1.0,      # 支持向量机模型的权重
    'rf': 1.0,       # 随机森林模型的权重
    'knn': 1.0,      # k最近邻模型的权重
    'naive_bayes': 1.0,  # 朴素贝叶斯模型的权重
    'dl': 3.5
}

# Assume all the input have the same shape (28, 28)
# Every func starts with predict_ returns an integer
def predict_sklearn(model, data):
    data = data.reshape(1, -1)  # Flatten the data for sklearn models
    return int(model.predict(data)[0])


def predict_torch(model, data):
    data = data.flatten()

    with torch.no_grad():
        output = model.predict_image_from_array(data)
        predicted = int(output)
    return predicted  # 返回一个整数


# 定义集成预测函数
def ensemble_predict(data):
    predictions = [
        ('log_reg', predict_sklearn(log_reg_model, data)),
        ('svm', predict_sklearn(svm_model, data)),
        ('rf', predict_sklearn(rf_model, data)),
        ('knn', predict_sklearn(knn_model, data)),
        ('naive_bayes', predict_sklearn(naive_bayes_model, data)),
        ('dl', predict_torch(torch_model, data))
    ]

    # Debugging output
    print(f"Predictions from all models with weights: {[(pred, weights[model]) for model, pred in predictions]}")

    # 加权投票
    vote_counts = defaultdict(float)
    for model, prediction in predictions:
        vote_counts[prediction] += weights[model]

    # 找出得票最高的预测结果
    final_prediction = max(vote_counts, key=vote_counts.get)
    return final_prediction


if __name__ == "__main__":
    X_sample = np.load('saved_array.npy')

    # Get the ensemble prediction
    final_prediction = ensemble_predict(X_sample)
    print("Ensemble Prediction:", final_prediction)