import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load the training dataset
train_data = pd.read_csv('sign_mnist_13bal_train.csv')

# Separate the data (features) and the classes
X_train_full = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train_full = X_train_full / 255.0
y_train_full = train_data['class']   # Target (first column)

# Split the training dataset into training and validation sets
X_train, X_validate, y_train, y_validate = train_test_split(
    X_train_full, y_train_full, test_size=20, random_state=2, stratify=y_train_full
)

# Load the validation dataset
validation_data = pd.read_csv('sign_mnist_13bal_test.csv')

# Separate the data (features) and the classes
X_validate_full = validation_data.drop('class', axis=1)  # Features (all columns except the first one)
X_validate_full = X_validate_full / 255.0
y_validate_full = validation_data['class']   # Target (first column)

hidden_layer_sizes = [(5,), (15,), (20,)]

for hl_size in hidden_layer_sizes:
    print(f'\nTesting hidden layer size: {hl_size}')
    # Adjust the neural network model design
    neural_net_model = MLPClassifier(hidden_layer_sizes=hl_size, random_state=42, tol=0.005)

    neural_net_model.fit(X_train, y_train)
    # Determine model architecture
    layer_sizes = [neural_net_model.coefs_[0].shape[0]]  # Start with the input layer size
    layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_]  # Add sizes of subsequent layers
    layer_size_str = " x ".join(map(str, layer_sizes))
    print(f"Layer sizes: {layer_size_str}")

    # Predict the classes from the training and validation sets
    y_pred_train = neural_net_model.predict(X_train)
    y_pred_validate = neural_net_model.predict(X_validate)

    # Calculate overall validation accuracy
    overall_accuracy_validate = (y_validate == y_pred_validate).mean() * 100
    print(f"Overall Validation Accuracy: {overall_accuracy_validate:3.1f}%")

    # Overall training accuracy
    overall_accuracy_train = (y_train == y_pred_train).mean() * 100
    print(f"Overall Training Accuracy: {overall_accuracy_train:3.1f}%")

    # Identify most frequently misidentified classes in the validation set
    misidentifications = defaultdict(int)
    for true, pred in zip(y_validate, y_pred_validate):
        if true != pred:
            misidentifications[true] += 1

    # Sort and print the most frequently misidentified classes
    misidentified_sorted = sorted(misidentifications.items(), key=lambda item: item[1], reverse=True)
    print("Most frequently misidentified classes and their counts:")
    for class_id, count in misidentified_sorted[:3]:
        print(f"Class {class_id} misidentified {count} times")