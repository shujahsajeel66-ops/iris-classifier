from sklearn.datasets import load_iris


def load():
    iris = load_iris(as_frame=True)
    data = iris.frame
    data["species"] = data["target"].map(dict(enumerate(iris.target_names)))
    return data


def explore(data):
    print(data.head(), "\n")
    print("Shape:", data.shape, "\n")
    print(data.isnull().sum(), "\n")
    print(data["species"].value_counts())


if __name__ == "__main__":
    data = load()
    explore(data)
    data.to_csv("data/iris.csv", index=False)
