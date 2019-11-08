import pandas as pd
import json

from cnn import cnn_runner
from svm import svm_runner


if __name__ == '__main__':
    Y_test = pd.read_csv("Y_test.csv")

    model, acc, roc = cnn_runner()
    preds = model.predict(Y_test)

    results = {"model": "CNN", "Accuracy": acc, "ROC": roc, "Predictions": preds }

    with open('CNN_predict_results.txt', 'w') as outfile:
        json.dump(results, outfile)