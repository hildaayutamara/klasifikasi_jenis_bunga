# Hermon Jay, 14-10-2017
# klasifikasi jenis bunga dengan
# SVM
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score

# model
svm = svm.SVC(C=2)

# data
# nama kolom = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_train = pd.read_csv('iris_training.csv', header=None)
iris_test = pd.read_csv('iris_test.csv', header=None)

# masukan X dan keluaran Y
X = iris_train.iloc[:,:4]
Y = iris_train.iloc[:,4]
X_test = iris_test.iloc[:,:4]
Y_test = iris_test.iloc[:,4]

# latih classifier
svm.fit(X,Y)

# prediksi data test
Y_pred = svm.predict(X_test)

# print persentase akurasi
akurasi = accuracy_score(Y_test, Y_pred)*100
print("Akurasi : %.2f" % akurasi, "%")