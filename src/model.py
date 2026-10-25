from sklearn.neighbors import KNeighborsClassifier

from explore import load
from preprocess import split, scale


def train(Xtrain, ytrain, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(Xtrain, ytrain)
    return model


if __name__ == "__main__":
    Xtrain, Xtest, ytrain, ytest = split(load())
    Xtrain, Xtest, scaler = scale(Xtrain, Xtest)

    model = train(Xtrain, ytrain)
    preds = model.predict(Xtest)

    print("Predicted:", preds)
    print("Actual:   ", ytest.values)
    print("Accuracy:", model.score(Xtest, ytest))
