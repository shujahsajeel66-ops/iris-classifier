from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from explore import load


def split(data):
    X = data.drop(columns=["target", "species"])
    y = data["target"]
    return train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )


def scale(Xtrain, Xtest):
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    return Xtrain, Xtest, scaler


if __name__ == "__main__":
    Xtrain, Xtest, ytrain, ytest = split(load())
    Xtrain, Xtest, scaler = scale(Xtrain, Xtest)
    print("Train:", Xtrain.shape, "Test:", Xtest.shape)
    print("Train mean:", Xtrain.mean(axis=0).round(2))
    print("Train std:", Xtrain.std(axis=0).round(2))
    print("Test mean:", Xtest.mean(axis=0).round(2))
