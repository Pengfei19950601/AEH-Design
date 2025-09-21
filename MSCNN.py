import os
import scipy.io
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Import confusion matrix functions

# Load data
Spectrum = scipy.io.loadmat(r'C:\Users\fanpe\Desktop\Spectrum10\Spectrum.mat')  # Load spectrum data
Spectrum1 = Spectrum['Spectrum']  # Extract spectrum data
Speed = scipy.io.loadmat(r'C:\Users\fanpe\Desktop\Spectrum10\Speed.mat')  # Load speed data
Speed1 = Speed['Speed']  # Extract speed data

# Check Speed1 data type and convert to numeric
if np.issubdtype(Speed1.dtype, np.number):
    # If already numeric, use directly
    Speed1_numeric = Speed1
else:
    # Otherwise, try to convert to float
    Speed1_numeric = Speed1.astype(float)

# Create new matrix Speed2
Speed2 = np.zeros(Speed1_numeric.shape, dtype=int)  # Initialize Speed2 as all-zero matrix

# Set Speed2 values
Speed2[Speed1_numeric == 30] = 0
Speed2[Speed1_numeric == 50] = 1
Speed2[Speed1_numeric == 70] = 2

# Prepare features and labels
X = Spectrum1  # Feature data maintains original shape
y = Speed2.flatten()  # Label data flattened to 1D array

# Convert labels to one-hot encoding
y_one_hot = tf.keras.utils.to_categorical(y, num_classes=3)  # Convert labels to one-hot encoding, 3 categories

# Data normalization
scaler = MinMaxScaler()  # Create normalizer
X_reshaped = X.reshape(X.shape[0], -1)  # Flatten data to 2D array
X_scaled = scaler.fit_transform(X_reshaped)  # Normalize data
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)  # Reshape to 3D array suitable for convolutional input

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=42)  # Split into train and test sets, 80% train, 20% test

# Define MultiScaleResNet model
class MultiScaleResNet(tf.keras.Model):
    def __init__(self):
        super(MultiScaleResNet, self).__init__()
        # Multi-scale convolutional layers
        self.conv1_3 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')  # 3x convolution
        self.conv1_5 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')  # 5x convolution
        self.conv1_7 = tf.keras.layers.Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')  # 7x convolution
        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)  # Max pooling layer
        # Second convolutional layer
        self.conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')  # 3x convolution
        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)  # Max pooling layer
        # Fully connected layers
        self.flatten = tf.keras.layers.Flatten()  # Flatten layer
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')  # Fully connected layer
        self.dropout = tf.keras.layers.Dropout(0.5)  # Dropout layer to prevent overfitting
        self.output_layer = tf.keras.layers.Dense(3, activation='softmax')  # Output layer, 3 categories

    def call(self, inputs):
        # Multi-scale convolution
        x3 = self.conv1_3(inputs)  # 3x convolution
        x5 = self.conv1_5(inputs)  # 5x convolution
        x7 = self.conv1_7(inputs)  # 7x convolution
        x = tf.concat([x3, x5, x7], axis=-1)  # Concatenate multi-scale features
        x = self.pool1(x)  # Pooling
        # Second convolutional layer
        x = self.conv2(x)  # 3x convolution
        x = self.pool2(x)  # Pooling
        # Fully connected layers
        x = self.flatten(x)  # Flatten
        x = self.dense1(x)  # Fully connected
        x = self.dropout(x)  # Dropout
        return self.output_layer(x)  # Output

# Create model instance
model = MultiScaleResNet()

# Compile model using AdamW optimizer
weight_decay = 1e-4  # Weight decay coefficient
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=weight_decay)  # Adam optimizer
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  # Compile model with cross-entropy loss

# Train model
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)  # Train model, 200 epochs, batch size 32, 20% as validation set

# Evaluate model on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)  # Evaluate model
print(f"Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy * 100:.2f}%")  # Print test loss and accuracy

# Generate confusion matrix
y_pred = model.predict(X_test)  # Predict
y_pred_classes = np.argmax(y_pred, axis=1)  # Get predicted classes
y_true_classes = np.argmax(y_test, axis=1)  # Get true classes

# Calculate confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)  # Calculate confusion matrix

# Print confusion matrix
print("Confusion Matrix:")
print(cm)

# Calculate accuracy for each class
accuracies = cm.diagonal() / cm.sum(axis=1)  # Accuracy for each class
for i, accuracy in enumerate(accuracies):
    print(f"Class {i} Accuracy: {accuracy * 100:.2f}%")  # Print accuracy for each class

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])  # Create confusion matrix display object
disp.plot(cmap='Blues')  # Plot confusion matrix