import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dl85 import DL85Classifier
import graphviz
import uuid


def get_dot_body(treedict, parent=None, left=True):
    gstring = ""
    id = str(uuid.uuid4())
    id = id.replace('-', '_')

    if "feat" in treedict.keys():
        feat = treedict["feat"]
        if parent is None:
            gstring += "node_" + id + " [label=\"{{feat|" + str(feat) + "}}\"];\n"
            gstring += get_dot_body(treedict["left"], id)
            gstring += get_dot_body(treedict["right"], id, False)
        else:
            gstring += "node_" + id + " [label=\"{{feat|" + str(feat) + "}}\"];\n"
            gstring += "node_" + parent + " -> node_" + id + " [label=" + str(int(left)) + "];\n"
            gstring += get_dot_body(treedict["left"], id)
            gstring += get_dot_body(treedict["right"], id, False)
    else:
        val = str(int(treedict["value"])) if treedict["value"] == int(treedict["value"]) else str(
            round(treedict["value"], 3))
        err = str(int(treedict["error"])) if treedict["error"] == int(treedict["error"]) else str(
            round(treedict["error"], 2))
        gstring += "leaf_" + id + " [label=\"{{class|" + val + "}|{error|" + err + "}}\"];\n"
        gstring += "node_" + parent + " -> leaf_" + id + " [label=" + str(int(left)) + "];\n"
    return gstring


def export_graphviz(tree_str):
    # initialize the header
    graph_string = "digraph Tree { \n" \
                   "graph [ranksep=0]; \n" \
                   "node [shape=record]; \n"

    # build the body
    graph_string += get_dot_body(tree_str)

    # end by the footer
    graph_string += "}"

    return graph_string


print("######################################################################\n"
      "#                      DL8.5 default classifier                      #\n"
      "######################################################################")

# read the dataset and split into features and targets
dataset = np.genfromtxt("../../datasets/anneal.txt", delimiter=' ')
X, y = dataset[:, 1:], dataset[:, 0]
# split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = DL85Classifier(max_depth=2, min_sup=1, print_output=True)
clf.fit(X, y)
print("========= END PRINT =========\n\n")
y_pred = clf.predict(X_test)

# show results
print("Model built in", round(clf.runtime_, 4), "seconds")
print("Found tree:", clf.tree_)
print("Confusion Matrix below\n", confusion_matrix(y_test, y_pred))
print("Accuracy on training set =", round(clf.accuracy_, 4))
print("Accuracy on test set =", round(accuracy_score(y_test, y_pred), 4))

# print the tree
dot = export_graphviz(clf.tree_)
graph = graphviz.Source(dot, format="png")
graph.render("plots/anneal_odt")
