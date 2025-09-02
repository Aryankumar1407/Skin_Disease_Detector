import os
import cv2   # Import OpenCV for image processing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Define labels for different skin diseases
labels_dict = {
    'Actinic keratosis': 0,
    'Atopic Dermatitis': 1,
    'Benign keratosis': 2,
    'Dermatofibroma': 3,
    'Melanocytic nevus': 4,
    'Melanoma': 5,
    'Squamous Cell Carcinoma': 6,
    'Tinea Ringworm Candidiasis': 7,
    'Vascular lesion': 8
}

# Define the directory containing the image data
data_dir = '/content/drive/MyDrive/Split_smol/train'

# Initialize empty lists to store image data and labels
X = []
y = []

# Iterate through each folder (representing different skin diseases) in the data directory
for folder, label in labels_dict.items():
    folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(folder_path):
        print(f"Directory '{folder}' not found in '{data_dir}'. Skipping...")
        continue

    # Iterate through each image in the folder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)  # Read the image using OpenCV
        if img is None:
            print(f"Failed to read image: {image_path}. Skipping...")
            continue

        img = cv2.resize(img, (227, 227))  # Resize images to (227, 227) for compatibility with InceptionV3
        X.append(img)  # Append the resized image to the list of images
        y.append(label)  # Append the corresponding label to the list of labels

# Convert the lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Shuffle the data to ensure randomness
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X_shuffled = X[indices]
y_shuffled = y[indices]

# Split the data into training and validation sets (80% training, 20% validation)
split_index = int(0.8 * len(X_shuffled))
X_train, X_val = X_shuffled[:split_index], X_shuffled[split_index:]
y_train, y_val = y_shuffled[:split_index], y_shuffled[split_index:]

# Data augmentation to increase the amount of available data
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create data generators for training and validation sets
train_generator = datagen.flow(X_train, y_train, batch_size=32)
val_generator = datagen.flow(X_val, y_val, batch_size=32)

# Load the InceptionV3 model with pre-trained weights
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(227, 227, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the pre-trained model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(labels_dict), activation='softmax')(x)

# Create the final model by specifying inputs and outputs
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with Adagrad optimizer and sparse categorical cross-entropy loss function
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01, initial_accumulator_value=0.1, epsilon=1e-07)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model on the training data
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator
)

# Load the test data and evaluate the model on it
test_dir = '/content/drive/MyDrive/Split_smol/val'
X_test = []
y_test = []

# Repeat the same data loading process as done for training and validation sets
# Here, we'll use the test data for evaluation
for folder, label in labels_dict.items():
    folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(folder_path):
        print(f"Directory '{folder}' not found in '{data_dir}'. Skipping...")
        continue

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}. Skipping...")
            continue

        img = cv2.resize(img, (227, 227))
        X_test.append(img)
        y_test.append(label)

# Convert the test data to numpy arrays
X_test = np.array(X)
y_test = np.array(y)

# Create a data generator for the test data
test_generator = datagen.flow(X_test, y_test, batch_size=32)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Epoch vs Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the trained model
model.save('skindisease_model_inceptionv3.h5')

# Load the saved model
new_model = tf.keras.models.load_model('skindisease_model_inceptionv3.h5')

# Load an example image for prediction
image_path = '/content/drive/MyDrive/Split_smol/val/Atopic Dermatitis/1_14.jpg'
test_image = PIL.Image.open(image_path)
resized_image = test_image.resize((227, 227))
resized_image = np.array(resized_image) / 255.0
resized_image = resized_image[np.newaxis, ...]

# Perform prediction using the loaded model
prediction = new_model.predict(resized_image)
predicted_class_index = np.argmax(prediction)
print('Predicted class index:', predicted_class_index)

# Retrieve the predicted label from the dictionary
predicted_label = {v: k for k, v in labels_dict.items()}.get(predicted_class_index)
print('Predicted label:', predicted_label)

# Print additional information about the predicted skin disease
if predicted_label:
    print('Predicted label:', predicted_label)
    info = disease_info.get(predicted_label)
    if info:
        print('Diagnosis:', info['diagnosis'])
        print('Severity:', info['severity'])
        print('Precautions:')
        for precaution in info['precautions']:
            print('-', precaution)
        print('Preventions:')
        for prevention in info['preventions']:
            print('-', prevention)
    else:
        print('Information not available for this disease.')
else:
    print('Unable to determine the predicted label.')
