// Fill-in-the-blank source code practice — the exact exercise type Oliver's exam
// announcement names ("there will be a fill in the blank source code question").
// Every snippet below is the real code from the script PDF (CI2__ML_with_sklearn-SHORT.pdf),
// with the key class, method or argument blanked out.
ML.fillcode = [
  {
    ch: "knn", title: "KNN classification pipeline on Iris", pts: 2,
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import __b1__
from sklearn.neighbors import __b2__
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, __b3__=y
)

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(__b4__=5)
)
model.__b5__(X_train, y_train)
predictions = model.__b6__(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))`,
    ans: { b1: "StandardScaler", b2: "KNeighborsClassifier", b3: "stratify", b4: "n_neighbors", b5: "fit", b6: "predict" },
  },
  {
    ch: "svm", title: "SVM with RBF kernel", pts: 2,
    code: `from sklearn.svm import __b1__

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="__b2__", C=1.0, __b3__="scale")
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)`,
    ans: { b1: "SVC", b2: "rbf", b3: "gamma" },
  },
  {
    ch: "forest", title: "Random Forest", pts: 2,
    code: `from sklearn.ensemble import __b1__

model = RandomForestClassifier(
    __b2__=100, random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)`,
    ans: { b1: "RandomForestClassifier", b2: "n_estimators" },
  },
  {
    ch: "lasso", title: "Lasso regression", pts: 2,
    code: `from sklearn.linear_model import __b1__

model = make_pipeline(StandardScaler(), Lasso(__b2__=1.0))
model.__b3__(X_train, y_train)
predictions = model.predict(X_test)
print("Weights:", model.named_steps["lasso"].__b4__)`,
    ans: { b1: "Lasso", b2: "alpha", b3: "fit", b4: "coef_" },
  },
  {
    ch: "poly", title: "Polynomial regression", pts: 2,
    code: `from sklearn.preprocessing import __b1__
from sklearn.linear_model import LinearRegression

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=__b2__),
    LinearRegression()
)`,
    ans: { b1: "PolynomialFeatures", b2: "False" },
  },
  {
    ch: "mlp", title: "MLP classifier", pts: 2,
    code: `from sklearn.neural_network import __b1__

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        __b2__=(20,),
        max_iter=2000,
        random_state=42
    )
)`,
    ans: { b1: "MLPClassifier", b2: "hidden_layer_sizes" },
  },
  {
    ch: "dbscan", title: "DBSCAN", pts: 2,
    code: `from sklearn.cluster import __b1__

labels = DBSCAN(__b2__=0.25, min_samples=5).__b3__(X)
n_clusters = len(set(labels) - {__b4__})`,
    ans: { b1: "DBSCAN", b2: "eps", b3: "fit_predict", b4: "-1" },
  },
  {
    ch: "pca", title: "PCA", pts: 2,
    code: `from sklearn.decomposition import __b1__

X_scaled = StandardScaler().__b2__(X)
pca = PCA(n_components=2)
Z = pca.__b3__(X_scaled)`,
    ans: { b1: "PCA", b2: "fit_transform", b3: "fit_transform" },
  },
  {
    ch: "mds", title: "MDS embedding", pts: 2,
    code: `from sklearn.manifold import __b1__

mds = MDS(n_components=2, n_init=4, random_state=42)
Z = mds.__b2__(X_scaled)`,
    ans: { b1: "MDS", b2: "fit_transform" },
  },
  {
    ch: "cv", title: "Cross-validated grid search", pts: 3,
    code: `from sklearn.model_selection import train_test_split, __b1__

model = make_pipeline(StandardScaler(), KNeighborsClassifier())
settings = {"kneighborsclassifier__b2__n_neighbors": [3, 5, 9]}

search = GridSearchCV(model, settings, cv=3, scoring="accuracy")
search.__b3__(X_train, y_train)

print("Best setting:", search.__b4__)
print("Test accuracy:", search.__b5__(X_test, y_test))`,
    ans: { b1: "GridSearchCV", b2: "__", b3: "fit", b4: "best_params_", b5: "score" },
  },
  {
    ch: "metrics", title: "Confusion matrix and report", pts: 1,
    code: `from sklearn.metrics import __b1__, classification_report

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, __b2__=0))`,
    ans: { b1: "confusion_matrix", b2: "zero_division" },
  },
  {
    ch: "linreg", title: "Linear Regression with evaluation metrics", pts: 2,
    code: `from sklearn.linear_model import __b1__
from sklearn.metrics import mean_absolute_error, __b2__
from sklearn.metrics import r2_score

model = LinearRegression()
model.__b3__(X_train, y_train)
predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)
print("R2:", __b4__(y_test, predictions))`,
    ans: { b1: "LinearRegression", b2: "mean_squared_error", b3: "fit", b4: "r2_score" },
  },
  {
    ch: "knn", title: "Stratified split reused for KNN", pts: 2,
    code: `X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, __b1__=y)

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.__b2__(X_test)))`,
    ans: { b1: "stratify", b2: "predict" },
  },
];
