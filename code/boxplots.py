import numpy as np
from utils.jaccard_similarity import jaccard_emb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from pyrdf2vec.graphs import KG
import rdflib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="aifb", help="Select the data: aifb, mutag, bgs")
parser.add_argument("--num_embeddings", type=str, default=10, help="Number of embeddings that should be generated")
args = parser.parse_args()

# preprocessing for the prediction task

# class labels are of the form "http://swrc.ontoware.org/ontology#Publiction"
# The following method strips the list of classes and returns it
def get_classes(data, classes_filt):
    classes = []
    for class_type in classes_filt:
        if data == "mutag":
            class_type = class_type.lstrip("http://dl-learner.org/carcinogenesis#")
            class_type = class_type.lstrip("www.w3.org/2002/07/owl#")
        else:
            class_type = class_type.lstrip("http://swrc.ontoware.org/ontology#")
            class_type = class_type.lstrip("3.org/2002/07/owl#")
        classes.append(class_type)
    return np.array(classes)

# Some classes occure only very rarely in the KG.
# The following method returns the most-occuring classes. We restrict the prediction task for only these classes.
def select_classes(data):
    if data == "aifb":
        return (["Publication", "Person", "InProceedings"])
    elif data == "mutag":
        return (["Hydrogen-3", "Bond-1", "Bond-7", "Carbon-22", "Carbon-10"])

# returns data (X,y) according to the classes selected by the select_class method
def select_data(X, data, classes_filt):
    classes = get_classes(data, classes_filt)
    selected_classes = select_classes(data)  # get the most prominent classes
    indexes = []
    for j in range(len(classes)):
        if classes[j] in selected_classes:
            indexes.append(j)
    X = X[indexes]
    y = classes[indexes]
    return (X, y)


def processing(data):
    # get the classes
    g = rdflib.Graph()
    if data == "aifb":
        g.parse("../data/aifbfixed_complete.n3", format="n3")
    elif data == "bgs":
        g.parse("../data/bgs.nt", format="nt")
    elif data == "mutag":
        g.parse("../data/mutag.xml", format="xml")

    q = g.query("""SELECT ?entity ?type
                    WHERE {
                      ?entity rdf:type ?type.
                    }""")
    entities = []
    classes = []
    for row in q:
        entities.append(row[0])
        classes.append(row[1])

    filtered_index = []
    index = 0
    for e in entities:
        if e in kg._entities:
            filtered_index.append(index)
        index += 1

    #entities_filt = [entities[i] for i in filtered_index]
    classes_filt = [classes[i] for i in filtered_index]


    num_embeddings = 10

    counter = 0
    results = np.zeros(2)
    for nlp_model in ("cbow", "skip-gram"):
        for dim_red in ("pca2d", "pca3d", "tsne2d", "tsne3d"):
            embeddings = []

            # read embeddings:
            if dim_red == "":
                for iter in range(num_embeddings):
                    embeddings.append(
                        np.load("embeddings/" + data + "/" + nlp_model + "/embedding" + str(iter) + ".npy"))
            else:
                for iter in range(num_embeddings):
                    embeddings.append(np.load(
                        "embeddings/" + data + "/" + nlp_model + "/" + dim_red + "/embedding" + str(iter) + ".npy"))

            # k-NN for each embedding
            for j in range(num_embeddings):
                X, y = select_data(embeddings[j], data, classes_filt)

                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                if j == 0:
                    df_y = pd.DataFrame(data=y_test, columns=["truth"])

                n_neighbors = 20
                clf = neighbors.KNeighborsClassifier(n_neighbors)

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                clf.fit(X_train, y_train)



                y_pred = clf.predict(X_test)
                # concatenate the columns
                df_y = pd.concat([df_y, pd.DataFrame(data=y_pred, columns=["Embedding" + str(j)])], axis=1)

            # calculate the pairwise accuracy
            y_same = np.zeros(len(y_test))
            for i in range(num_embeddings - 1):
                for j in range(i + 1, num_embeddings):
                    y_same += (df_y["Embedding" + str(i)] == df_y["Embedding" + str(j)])

            if counter == 0:
                results = pd.DataFrame({"accuracy": y_same / (num_embeddings * (num_embeddings - 1) / 2),
                                        "method": np.repeat(nlp_model, len(y_same)),
                                        "dim_red": np.repeat(dim_red, len(y_same))},
                                       columns=("accuracy", "method", "dim_red"))
            else:
                results = results.append(pd.DataFrame({"accuracy": y_same / (num_embeddings * (num_embeddings - 1) / 2),
                                                       "method": np.repeat(nlp_model, len(y_same)),
                                                       "dim_red": np.repeat(dim_red, len(y_same))},
                                                      columns=("accuracy", "method", "dim_red")))
            counter += 1

    return (results)

if __name__ == "__main__":
    ###########################################################
    # 1. Boxplot that compares all methods for data args.data #
    ###########################################################
    data = args.data
    num_embeddings = args.num_embeddings
    embeddings = []
    len_emb = len(np.load("embeddings/" + data + "/cbow/pca2d/embedding1.npy"))

    results = np.zeros((1, 2))
    i = 0
    for nlp_model in ("cbow", "skip-gram"):
        for dim_red in ("pca2d", "pca3d", "tsne2d", "tsne3d"):
            embeddings = []
            for iter in range(num_embeddings):
                embeddings.append(np.load("embeddings/" + data + "/" + nlp_model + "/" + dim_red + "/embedding" + str(iter) + ".npy"))

            dist = jaccard_emb(embeddings, pairwise=True)
            if i==0:
                results = pd.DataFrame({"distance":dist, "method":np.repeat(nlp_model, len_emb), "dim_red": np.repeat(dim_red, len_emb)}, columns=("distance", "method", "dim_red") )
            else:
                results = results.append(pd.DataFrame({"distance":dist, "method":np.repeat(nlp_model, len_emb), "dim_red": np.repeat(dim_red, len_emb)}, columns=("distance", "method", "dim_red") ))
            i += 1


    # generate boxplot
    PATH = "scores/" + data + "/plots/"
    fig, ax = plt.subplots()
    color = sns.color_palette("tab10")
    ax = sns.boxplot(x="dim_red", y="distance", hue="method", data=results, palette="tab10")
    ax.set_xlabel("Dimension reduction", fontsize=20, weight="bold")
    ax.set_ylabel("Pairwise Jaccard dist.", fontsize=20, weight="bold")
    ax.tick_params(labelsize=15)
    plt.legend(loc='upper left', fontsize=15)
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.suptitle("Pairwise similarities: " + data, fontsize=30, weight="bold")
    fig.savefig(PATH + "comparison_boxplot.pdf")
    plt.show()
    print("The figures are saved in:" + PATH + " (comparison_boxplot.pdf and prediction_boxplot.pdf)")




    ######################################
    # 2. Compare the prediction results: #
    ######################################
    if data == "bgs":
        print("Prediction results are not available for bgs data")
    else:
        if data == "aifb":
            kg = KG(location="../data/aifbfixed_complete.n3", file_type="n3")
        elif data == "bgs":
            kg = KG(location="../data/bgs.nt", file_type="nt")
        elif data== "mutag":
            kg = KG(location="../data/mutag.xml", file_type="xml")

        results = processing(data)

        fig, ax = plt.subplots()
        sns.set_style("white")
        color = sns.color_palette("tab10")
        ax = sns.boxplot(x="dim_red", y="accuracy", hue="method", data=results, palette="tab10")
        #ax.set_title("Pairwise similarities: " + data, fonzsize=30)
        ax.set_xlabel("Dimension reduction", fontsize=20, weight="bold")
        ax.set_ylabel("% same prediction", fontsize=20, weight="bold")
        fig.suptitle("Similar predictions: " + data, fontsize=30, weight="bold")
        plt.legend(loc='upper left', fontsize=15)
        ax.tick_params(labelsize=15)
        plt.gcf().subplots_adjust(bottom=0.15)
        fig.savefig(PATH + "prediction_boxplot.pdf")
        plt.show()