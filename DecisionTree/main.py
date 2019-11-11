import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix

dataset=pd.read_csv('divorce.csv',delimiter=";")


a=dataset.drop_duplicates()
print("DUPLICATE SONRASI YENİ VERİ SAYIMIZ:")
print(len(a))

X=dataset.iloc[:,0:54]
y=dataset["Class"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=1)
#PREPROCESSING YAPILARAK TEKRARLI VERİLER SİLİNDİ VE TEST TRAIN DATALARI AYRILDI

clf = DecisionTreeClassifier()
#SINIFLANDIRMADAN ONCE DATALARIMIZI CROSS VALIDATION YONTEMİYLE KOMBİNASYON OLUŞTURUYORUZ. 10 PARÇAYA BÖLÜNÜYOR
cv_results = cross_validate(estimator=clf, X=X, y=y.values.ravel(), cv=10)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)

print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))






#PRECIOSN SCORE
from sklearn.metrics import precision_score, roc_auc_score

print("Precision")
print(precision_score(y_test, y_pred, average='weighted'))



#RECALL PRECISON İLE AYNI

from sklearn.metrics import recall_score, roc_auc_score
print("Recall")
print(recall_score(y_test, y_pred, average='weighted'))

print("F-MEAUSRE")
precision=precision_score(y_test, y_pred, average='weighted')
recall=recall_score(y_test, y_pred, average='weighted')
fm=2*(precision*recall/(precision+recall))
print(fm)




def plot_roc_curve(rfpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

data_X, class_label = make_classification(n_samples=1000, n_classes=2, weights=[1, 1], random_state=1)

trainX, testX, trainy, testy = train_test_split(data_X, class_label, test_size=0.3, random_state=1)

model = RandomForestClassifier()
model.fit(trainX, trainy)

probs = model.predict_proba(testX)

probs = probs[:, 1]

auc = roc_auc_score(testy, probs)
print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(testy, probs)

plot_roc_curve(fpr, tpr)



print('Standard Deviation:', np.std(dataset))
