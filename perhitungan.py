import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Langkah 1: Memuat data dari file Excel
file_path = 'Data TB 987 record.xlsx'  # Sesuaikan dengan path file Anda
data = pd.read_excel(file_path)

# Memeriksa nama-nama kolom dalam dataset
print("Nama-nama kolom dalam dataset:", data.columns)

# Lanjutkan jika nama kolom sudah dipastikan benar
label_column = 'LOKASI ANATOMI (target/output)'  # Sesuaikan dengan nama kolom yang benar

# Langkah 2: Pra-pemrosesan data
# Mengonversi label kategorikal ke numerik
label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])

# Memisahkan fitur dan label
X = data.drop(label_column, axis=1)
y = data[label_column]

# Fungsi untuk menghitung entropy
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

# Fungsi untuk menghitung gain
def information_gain(X, y, feature):
    total_entropy = entropy(y)
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(y[X[feature] == values[i]]) for i in range(len(values))])
    return total_entropy - weighted_entropy

# Fungsi untuk menghitung split info
def split_info(X, feature):
    values, counts = np.unique(X[feature], return_counts=True)
    probabilities = counts / len(X)
    return -np.sum(probabilities * np.log2(probabilities))

# Fungsi untuk menghitung gain ratio
def gain_ratio(X, y, feature):
    gain = information_gain(X, y, feature)
    split_info_value = split_info(X, feature)
    if split_info_value == 0:
        return 0  # Avoid division by zero
    return gain / split_info_value

# Menghitung entropy, gain, split info, dan gain ratio untuk setiap fitur
for feature in X.columns:
    print(f'Feature: {feature}')
    print(f'  Entropy: {entropy(y):.4f}')
    print(f'  Information Gain: {information_gain(X, y, feature):.4f}')
    print(f'  Split Info: {split_info(X, feature):.4f}')
    print(f'  Gain Ratio: {gain_ratio(X, y, feature):.4f}')
    print('')
