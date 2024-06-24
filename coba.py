import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
import graphviz

# Langkah 1: Memuat data dari file Excel
file_path = 'Data TB 987 record.xlsx'  # Sesuaikan dengan path file Anda
data = pd.read_excel(file_path)

# Memeriksa nama-nama kolom dalam dataset
print("Nama-nama kolom dalam dataset:", data.columns)

# Menggunakan nama kolom yang benar untuk target/output
label_column = 'LOKASI ANATOMI (target/output)'  # Sesuaikan dengan nama kolom yang benar

# Langkah 2: Pra-pemrosesan data
# Mengonversi label kategorikal ke numerik
label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])

# Memisahkan fitur dan label
X = data.drop(label_column, axis=1)
y = data[label_column]

# Langkah 3: Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Langkah 4: Membangun dan melatih model decision tree
clf = DecisionTreeClassifier(criterion='entropy')  # Menggunakan 'entropy' sebagai alternatif C4.5
clf = clf.fit(X_train, y_train)

# Prediksi dengan data testing
y_pred = clf.predict(X_test)

# Evaluasi model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

# Langkah 5: Visualisasi decision tree
feature_names = X.columns
class_names = label_encoder.classes_

dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=feature_names,  
                           class_names=class_names,  
                           filled=True, rounded=True,  
                           special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("decision_tree")  # Menyimpan visualisasi ke file 'decision_tree'

# Untuk menampilkan graph di Jupyter Notebook, gunakan:
# graph
