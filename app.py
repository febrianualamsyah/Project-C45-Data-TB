from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(X, y, feature):
    total_entropy = entropy(y)
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(y[X[feature] == values[i]]) for i in range(len(values))])
    return total_entropy - weighted_entropy

def split_info(X, feature):
    values, counts = np.unique(X[feature], return_counts=True)
    probabilities = counts / len(X)
    return -np.sum(probabilities * np.log2(probabilities))

def gain_ratio(X, y, feature):
    gain = information_gain(X, y, feature)
    split_info_value = split_info(X, feature)
    if split_info_value == 0:
        return 0  # Avoid division by zero
    return gain / split_info_value

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = 'uploads/' + file.filename
            file.save(file_path)

            # Process the uploaded file
            data = pd.read_excel(file_path)

            # Assuming the label_column is already known or specified
            label_column = 'LOKASI ANATOMI (target/output)'  # Adjust this to your label column

            label_encoder = LabelEncoder()
            for column in data.columns:
                data[column] = label_encoder.fit_transform(data[column])

            X = data.drop(label_column, axis=1)
            y = data[label_column]

            # Calculate gain ratio for each feature
            results = []
            max_gain_ratio = -float('inf')
            best_feature = None

            for feature in X.columns:
                current_gain_ratio = gain_ratio(X, y, feature)
                results.append({
                    'feature': feature,
                    'entropy': f'{entropy(y):.4f}',
                    'information_gain': f'{information_gain(X, y, feature):.4f}',
                    'split_info': f'{split_info(X, feature):.4f}',
                    'gain_ratio': f'{current_gain_ratio:.4f}'
                })

                # Update max gain ratio and best feature
                if current_gain_ratio > max_gain_ratio:
                    max_gain_ratio = current_gain_ratio
                    best_feature = feature

            # Render template with results, best feature, and max gain ratio
            return render_template('index.html', results=results, best_feature=best_feature, max_gain_ratio=f'{max_gain_ratio:.4f}')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
