import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json

# Paths
RAW_DATA_DIR = 'data/raw_data'          # Your original data folder with subfolders per class
PROCESSED_DATA_DIR = 'data/processed'   # We'll split data here into train/val folders
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'pattern_classifier_transfer.h5')
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, 'class_indices.json')

# Parameters
IMG_WIDTH, IMG_HEIGHT = 128, 128
BATCH_SIZE = 32
EPOCHS_FROZEN = 10       # Train with frozen base first
EPOCHS_FINE_TUNE = 10    # Then fine-tune last layers

def prepare_data():
    # Clear processed dir if exists
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)

    classes = os.listdir(RAW_DATA_DIR)
    for class_name in classes:
        images = os.listdir(os.path.join(RAW_DATA_DIR, class_name))
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        for category, imgs in [('train', train_imgs), ('val', val_imgs)]:
            category_path = os.path.join(PROCESSED_DATA_DIR, category, class_name)
            os.makedirs(category_path, exist_ok=True)
            for img_name in imgs:
                src = os.path.join(RAW_DATA_DIR, class_name, img_name)
                dst = os.path.join(category_path, img_name)
                shutil.copy(src, dst)

def create_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'train'),
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'val'),
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator

def build_transfer_model(num_classes):
    base_model = MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model, base_model

def fine_tune_model(model, base_model, train_generator, val_generator):
    # Unfreeze some layers of the base model
    base_model.trainable = True

    # Freeze all layers except last 50 layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Starting fine-tuning...")
    model.fit(
        train_generator,
        epochs=EPOCHS_FINE_TUNE,
        validation_data=val_generator,
        verbose=1
    )

def save_class_indices(train_generator):
    class_indices = train_generator.class_indices
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(class_indices, f)
    print(f"Class indices saved to {CLASS_INDICES_PATH}")

def train():
    print("Preparing data...")
    prepare_data()

    print("Creating data generators...")
    train_generator, val_generator = create_generators()

    print("Building model...")
    model, base_model = build_transfer_model(num_classes=train_generator.num_classes)

    print("Training with frozen base model...")
    model.fit(
        train_generator,
        epochs=EPOCHS_FROZEN,
        validation_data=val_generator,
        verbose=1
    )

    fine_tune_model(model, base_model, train_generator, val_generator)

    print("Saving model...")
    model.save(MODEL_PATH)
    save_class_indices(train_generator)
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train()
