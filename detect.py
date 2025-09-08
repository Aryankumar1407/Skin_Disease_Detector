import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from PIL import Image
import argparse

# ------------------------------
# Command line arguments
# ------------------------------
parser = argparse.ArgumentParser(description="Skin Disease Detection")
parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True,
                    help="Mode: train or predict")
parser.add_argument("--image", type=str, help="Path to image for prediction (required in predict mode)")
parser.add_argument("--force", action="store_true",
                    help="Force retraining even if model already exists")
args = parser.parse_args()

# ------------------------------
# Paths
# ------------------------------
train_dir = "/Users/aryankumar17/Downloads/skin_dataset/train"
val_dir   = "/Users/aryankumar17/Downloads/skin_dataset/val"
MODEL_PATH = "skindisease_model_inceptionv3.keras"

# ------------------------------
# Labels
# ------------------------------
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
idx_to_label = {v: k for k, v in labels_dict.items()}

# ------------------------------
# TRAIN MODE
# ------------------------------
if args.mode == "train":
    if os.path.exists(MODEL_PATH) and not args.force:
        print(f"✅ Model already exists at {MODEL_PATH}. Skipping training.")
        print("👉 Use --force to retrain anyway.")
    else:
        print("🔹 Training model...")

        # Load training data
        X, y = [], []
        for folder, label in labels_dict.items():
            folder_path = os.path.join(train_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                img = cv2.imread(image_path)
                if img is None:
                    continue
                img = cv2.resize(img, (227, 227))
                X.append(img)
                y.append(label)

        X = np.array(X) / 255.0
        y = np.array(y)

        # Shuffle & split
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Generators
        datagen = ImageDataGenerator()
        train_gen = datagen.flow(X_train, y_train, batch_size=32)
        val_gen = datagen.flow(X_val, y_val, batch_size=32)

        # Build model
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(227, 227, 3))
        for layer in base_model.layers:
            layer.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(len(labels_dict), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adagrad(0.01),
                      metrics=['accuracy'])

        # Train
        history = model.fit(train_gen, epochs=10, validation_data=val_gen)

        # Save
        model.save(MODEL_PATH)
        print(f"✅ Model trained and saved to {MODEL_PATH}")

# ------------------------------
# PREDICT MODE
# ------------------------------
elif args.mode == "predict":
    if not args.image:
        print("❌ Please provide an image path with --image for prediction.")
        exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}. Please run training first.")
        exit(1)

    print(f"🔹 Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"🔍 Analyzing image: {args.image}")
    img = Image.open(args.image).resize((227, 227))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    predicted_label = idx_to_label[class_idx]
    confidence = prediction[0][class_idx] * 100

    print("✅ Prediction complete")
    print(f"Predicted label: {predicted_label} (confidence: {confidence:.2f}%)")

    # Show probabilities for all classes
    print("\n📊 Prediction probabilities:")
    for idx, prob in sorted(enumerate(prediction[0]), key=lambda x: x[1], reverse=True):
        print(f"- {idx_to_label[idx]}: {prob*100:.2f}%")
