from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# load data
data = datasets.load_breast_cancer()
X = data.data[:, :2]  # first 2 columns for easier plotting
y = data.target

# scale features
sc = StandardScaler()
X = sc.fit_transform(X)

# split data
xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

# linear SVM
clf1 = SVC(kernel='linear', C=1)
clf1.fit(xtr, ytr)

# RBF SVM
clf2 = SVC(kernel='rbf', C=1, gamma='scale')
clf2.fit(xtr, ytr)

# accuracy with cross-validation
print("Linear acc:", cross_val_score(clf1, X, y, cv=5).mean())
print("RBF acc:", cross_val_score(clf2, X, y, cv=5).mean())

# plotting decision boundary function
def plot(model, X, y, title):
    h = 0.02
    xm, xM = X[:,0].min()-1, X[:,0].max()+1
    ym, yM = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(xm, xM, h), np.arange(ym, yM, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.show()

plot(clf1, X, y, "Linear SVM")
plot(clf2, X, y, "RBF SVM")
