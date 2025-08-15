## SVM Linear and RBF ##


This is a simple program to check how Support Vector Machine works with linear and RBF kernels.
I used the breast cancer dataset from sklearn and picked only the first 2 features so I can plot the decision boundaries easily.

## Steps

Load dataset

Scale the features

Split into train and test sets

Train SVM with linear kernel

Train SVM with RBF kernel

Check accuracy with cross-validation

Plot decision boundaries for both models


## Results

Linear kernel accuracy: ~94%

RBF kernel accuracy: ~95%
(Accuracy can change each time you run it)

## ![1000020016](https://github.com/user-attachments/assets/eca419df-3c90-40b6-b804-f3a2cbacbb84)



## What I learned

Support vectors are the points close to the decision boundary

C parameter controls the margin strictness

Kernels change data into another form to make separation easier

Linear kernel makes a straight line boundary

RBF can make curved boundaries for non-linear data

SVM can also be used for regression (SVR)

Good parameter selection avoids overfitting



## How to run

Install:

pip install scikit-learn matplotlib numpy

Run:

python svm_task.py
