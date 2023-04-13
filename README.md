# **Metode Klasifikas**
## **1. Read Data Set**

Quantitative Attributes:

1. x1 = Age (years)
2. x2 = BMI (kg/m2)
3. x3 = Glucose (mg/dL)
4. x4 = Insulin (µU/mL)
5. x5 = HOMA
6. x6 = Leptin (ng/mL)
7. x7 = Adiponectin (µg/mL)
8. x8 = Resistin (ng/mL)
9. x9 = MCP-1(pg/dL)

Classification (y):
1. Healthy controls
2. Patients

reference:

Patrício, M., Pereira, J., Crisóstomo, J., Matafome, P., Gomes, M., Seiça, R., & Caramelo, F. (2018). Using Resistin, glucose, age and BMI to predict the presence of breast cancer. BMC Cancer, 18(1).

### 1.1 Import library
```python
import pandas as pd
import numpy as np
```

### 1.2 Menampilkan Data Set
```python
data_olah = pd.read_csv('Data\dataR2.csv')
data_olah.head()
```
output:

![image](https://user-images.githubusercontent.com/117576737/231871013-0ae3516d-3291-4872-b97c-89974bfda0e8.png)

### 1.3 Split Data
```python
from sklearn.model_selection import train_test_split
X = data_olah.iloc[:,:-1]
y = data_olah.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print('Jumlah data train:',len(X_train))
print('Jumlah data test :',len(X_test))
```
output:
```
Jumlah data train: 87
Jumlah data test : 29
```
### 1.4 Visualisasi Data
```python
import seaborn as sn
import matplotlib.pyplot as plt

## Set theme ##
sn.set_theme(style='darkgrid')
```
```python
## Visualisasi Data X ##
data_olah.iloc[:,:-1].hist(figsize=(10,10), color='#01b1b5')
plt.show()
```
output:

![image](https://user-images.githubusercontent.com/117576737/231871840-ca527c1a-f331-4162-942f-69b52f9f58c3.png)
![image](https://user-images.githubusercontent.com/117576737/231871912-6dcdb562-c4a4-49ce-94b8-3b52c275b5f8.png)
![image](https://user-images.githubusercontent.com/117576737/231872015-4a251425-9243-49f7-8dd4-62760d9af142.png)

```python
## Visualisasi Data Y

## membuat variabel baru Y untuk menyimpan uniq dari y ##
Y = []
for i in pd.unique(y):
    Y.append(i)
    
nY = []
tmp0 = 0; #index#
for i in Y:
    nY.append(0)
    for j in y:
        if j == i:
            nY[tmp0] += 1    
    tmp0 += 1
```
```python
names = ['Normal', 'Beresiko']
marks = nY
my_circle = plt.Circle((0, 0), 0.6, color='white')
plt.pie(marks, labels=names, autopct='%.2f%%', colors=['#01b1b5', '#FF8080'])
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
```
output:

![image](https://user-images.githubusercontent.com/117576737/231872252-e13ffb2e-e2cc-4976-adf6-ac4c4b251f82.png)
```python
## Visualisasi Hubungan Umur dengan Variabel lain ##

tmp0 = 0
for i in data_olah.columns:
    if (tmp0 > 0) and (tmp0 < len(data_olah.columns)-1):
        ax = data_olah[data_olah['Classification'] == 1].plot(kind='scatter',
                                                      x='Age', 
                                                      y=i,
                                                      color='#01b1b5', 
                                                      label='Normal');
        data_olah[data_olah['Classification'] == 2].plot(kind='scatter', 
                                                         x='Age', 
                                                         y=i, 
                                                         color='#FF8080', 
                                                         label='Beresiko', 
                                                         ax=ax);
        plt.show()
    tmp0 += 1
```
output:

![image](https://user-images.githubusercontent.com/117576737/231872548-1ede6288-b2b3-497d-9ff6-2edd76a9d61f.png)
![image](https://user-images.githubusercontent.com/117576737/231872500-23425d87-b1b7-4da8-9b66-339174b5734c.png)
![image](https://user-images.githubusercontent.com/117576737/231872583-4d9ca79e-8189-4ef2-94dd-befcb8cfb1c8.png)
![image](https://user-images.githubusercontent.com/117576737/231872639-eb885728-9529-4e1d-8893-2777c625f031.png)
![image](https://user-images.githubusercontent.com/117576737/231872676-d9e0df38-dbb4-4188-9b99-84a63b621395.png)
![image](https://user-images.githubusercontent.com/117576737/231872731-92b70ab8-439e-4de7-bd59-59fa176734c8.png)
![image](https://user-images.githubusercontent.com/117576737/231872773-9141d8ae-e308-42ec-b4d6-80335bab9734.png)
![image](https://user-images.githubusercontent.com/117576737/231872814-fa304a2b-2163-4510-a76c-e6cffd79c2e8.png)

## **2. Metode Klasifikasi**
### 2.1 k – Nearest Neighbor (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
```
```python
knn_method = KNeighborsClassifier()
knn_method.fit(X_train, y_train)
y_pred_knn = knn_method.predict(X_test)
score_knn = metrics.accuracy_score(y_pred_knn, y_test)
print('Akurasi Prediksi method K-NN : ',round(score_knn,8), ' atau ',round(score_knn*100,2),'%', sep='')
```
output:
```
Akurasi Prediksi method K-NN : 0.65517241 atau 65.52%
```
```python
## Evaluasi model ##
k = 10
mean_acc = []
std_acc = []

for i in range(1,k):
    knn_method = KNeighborsClassifier(n_neighbors = i)
    knn_method.fit(X_train, y_train)
    y_pred_knn = knn_method.predict(X_test)
    mean_acc.append(metrics.accuracy_score(y_pred_knn, y_test))
    std_acc.append(np.std(y_pred_knn==y_test)/np.sqrt(y_pred_knn.shape[0]))
    
mean_acc
```
output:
```
[0.5172413793103449,
 0.6551724137931034,
 0.5517241379310345,
 0.3793103448275862,
 0.6551724137931034,
 0.5517241379310345,
 0.4827586206896552,
 0.4482758620689655,
 0.5862068965517241]
 ```
 ```python
 ## K-NN ##
plt.figure(figsize = (10,5))
plt.plot(range(1,k),mean_acc, color='#01b1b5')
plt.fill_between(range(1,k),
                 np.array(mean_acc) - 1 * np.array(std_acc),
                 mean_acc + 1 * np.array(std_acc), 
                 alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Nilai K')
plt.tight_layout()
plt.show()
```
output:

![image](https://user-images.githubusercontent.com/117576737/231873260-9be7b730-230c-44f6-889e-d6578df752b9.png)
```python
print("Nilai Akurasi terbaik ada pada ", 
      round(np.array(mean_acc).max()*100,2), 
      "% dengan k = ", np.array(mean_acc).argmax()+1, sep='') 
```
output:
```
Nilai Akurasi terbaik ada pada 65.52% dengan k = 2
```
### 2.2 Desicion Tree
```python
from sklearn.tree import DecisionTreeClassifier
```
```python
tree_method = DecisionTreeClassifier()
tree_method.fit(X_train, y_train)
y_pred_tree = tree_method.predict(X_test)
score_tree = metrics.accuracy_score(y_pred_tree, y_test)
print('Akurasi Prediksi method Tree : ',round(score_tree,8), ' atau ',round(score_tree*100,2),'%', sep='')
```
output:
```
Akurasi Prediksi method Tree : 0.75862069 atau 75.86%
```
```python
## Evaluasi Model ##

from sklearn import tree
plt.figure(figsize=(20,20))
tree.plot_tree(tree_method)
plt.show()
```
output:

![image](https://user-images.githubusercontent.com/117576737/231873675-6e1cc9b4-2702-4fd2-98d1-632ac0d9e19a.png)
### 2.3 Nive Bayes
```python
from sklearn import naive_bayes
```
```python
bayas_method = naive_bayes.BernoulliNB()
bayas_method.fit(X_train, y_train)
y_pred_bayes = bayas_method.predict(X_test)
score_bayes = metrics.accuracy_score(y_pred_bayes, y_test)
print('Akurasi Prediksi Bayes : ',round(score_bayes,8), ' atau ',round(score_bayes*100,2),'%', sep='')
```
output:
```
Akurasi Prediksi Bayes : 0.62068966 atau 62.07%
```
## **3. Akusisi Data**
### 3.1 Membuat Fungsi Confusion Matriks
```python
def get_conf_matriks(y_actual, y_predic, cmap = None, title='Tidak ada', ):
    confusion_matrix = metrics.confusion_matrix(y_actual, y_predic)
    ax = sn.heatmap(confusion_matrix, annot=True, fmt='g', cmap = cmap)
    ax.set_xlabel("Prediksi", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(['Salah', 'Benar'])
    ax.set_ylabel("Actual", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(['Salah', 'Benar'])
    ax.set_title(title, fontsize=14, pad=20)
    plt.show()
    
    conf_matrix_value = (confusion_matrix[0,0] + confusion_matrix[1,1]) / sum(sum(confusion_matrix))
    print("Hasil Confution Matriks :" , round(conf_matrix_value*100,2), "%", sep='')
```
### 3.2 Matriks Confution K-NN
```python
get_conf_matriks(y_test, y_pred_knn, title='Matriks Confution K-NN')
```
output:

![image](https://user-images.githubusercontent.com/117576737/231874171-d35e3330-96f5-43e2-818a-c5257adaed89.png)
```
Hasil Confution Matriks :58.62%
```
### 3.3 Matriks Confution Decition Tree
```python
get_conf_matriks(y_test, y_pred_tree, title='Matriks Confution Decition Tree')
```
output:

![image](https://user-images.githubusercontent.com/117576737/231874353-ad740906-88ec-4bea-a968-1e856ef6f6f1.png)
```
Hasil Confution Matriks :75.86%
```
### 3.4 Matriks Confution Nive Bayes
```python
get_conf_matriks(y_test, y_pred_bayes, title='Matriks Confution Nive Bayes')
```
output:

![image](https://user-images.githubusercontent.com/117576737/231874469-1240ce86-c089-4f40-9de5-6f8e547efe34.png)

```
Hasil Confution Matriks :62.07%
```
## **4. Kesimpulan**
Dari ketiga metode yang digunakan yaitu K-NN, Decitioin Tree, dan Nive Bayes untuk melakukan clasifiakasi pada data set dengan harapan ditemukan orang yang beresiko kanker atau tidak (normal), dengan melihat Akurasi dan hasil dari Confution Matriks dari ketiga metode. Maka Metode yang terbaik untuk melakukan klasifikasi pada data set yaitu Devition Tree dengan nilai akurasi 75.86%. atau dapat di lihat matriks confution dari Devition Tree berikut:
```python
get_conf_matriks(y_test, y_pred_tree, title='Matriks Confution Decition Tree')
```
output:

![image](https://user-images.githubusercontent.com/117576737/231874353-ad740906-88ec-4bea-a968-1e856ef6f6f1.png)
```
Hasil Confution Matriks :75.86%
```
