import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from explore import load
from preprocess import split, scale


def search(Xtrain, ytrain, ks):
    errors = []
    for k in ks:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, Xtrain, ytrain, cv=5)
        errors.append(1 - scores.mean())
    return errors


if __name__ == "__main__":
    Xtrain, Xtest, ytrain, ytest = split(load())
    Xtrain, Xtest, scaler = scale(Xtrain, Xtest)

    ks = list(range(1, 31))
    errors = search(Xtrain, ytrain, ks)
    print("Best K:", ks[errors.index(min(errors))])

    plt.plot(ks, errors, marker="o")
    plt.xlabel("K")
    plt.ylabel("Error rate")
    plt.title("Choosing K")
    plt.savefig("images/kvalue.png")
