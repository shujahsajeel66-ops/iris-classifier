import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from explore import load
from preprocess import split, scale
from model import train

names = ["setosa", "versicolor", "virginica"]


def evaluate(model, Xtest, ytest):
    preds = model.predict(Xtest)
    matrix = confusion_matrix(ytest, preds)
    print(matrix, "\n")
    print(classification_report(ytest, preds, target_names=names))
    return matrix


def plot(matrix):
    sns.heatmap(matrix, annot=True, cmap="Blues",
                xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("images/confusion.png", bbox_inches="tight")


if __name__ == "__main__":
    Xtrain, Xtest, ytrain, ytest = split(load())
    Xtrain, Xtest, scaler = scale(Xtrain, Xtest)
    model = train(Xtrain, ytrain, k=5)
    plot(evaluate(model, Xtest, ytest))
