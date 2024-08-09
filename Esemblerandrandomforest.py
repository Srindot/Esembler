from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score

print("Loadig Iris Dataset")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Logistic Regression")
log_clf = LogisticRegression(max_iter=1000)
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1
)

bag_clf.fit(X_train, y_train)
y_pred_bag = bag_clf.predict(X_test)
print("Bagging Classifier:", accuracy_score(y_test, y_pred_bag))

# Create and train the VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard'
)
voting_clf.fit(X_train, y_train)

# Train and evaluate each classifier
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Comparing the results between different classifiers vs voting classfier")
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

print("Random Forest Classifier")
rnd_clf_500 = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf_500.fit(X_train, y_train)
y_pred_rf = rnd_clf_500.predict(X_test)
print("Random Forest (500 trees, max leaf nodes=16):", accuracy_score(y_test, y_pred_rf))


rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])
print("\nFeature importances from RandomForestClassifier:")
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

print("Adaptive boost classifier")
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5
)
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)
print("AdaBoost Classifier:", accuracy_score(y_test, y_pred_ada))

print("Decission tree modifier")
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, y)
y2 = y - tree_reg1.predict(X)

tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X, y2)
y3 = y2 - tree_reg2.predict(X)

tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)

print("sum modifier")
y_pred = sum(tree.predict(X) for tree in (tree_reg1, tree_reg2, tree_reg3))

print("\nDecision Tree Regressor ensemble prediction:", y_pred)
