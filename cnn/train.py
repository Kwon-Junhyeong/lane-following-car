import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

train_dir = "train_data"
test_dir = "test_data"
img_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("[INFO] loading data")
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)
print(f"[INFO] train: {train_gen.samples} / test: {test_gen.samples}")

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 15
print("[INFO] training")
for epoch in range(epochs):
    print(f"\n========== Epoch {epoch+1}/{epochs} ==========")
    pbar = tqdm(total=len(train_gen), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    for step, (x_batch, y_batch) in enumerate(train_gen):
        loss, acc = model.train_on_batch(x_batch, y_batch)
        pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{acc:.4f}"})
        pbar.update(1)
        if step >= len(train_gen) - 1:
            break
    pbar.close()
    val_loss, val_acc = model.evaluate(test_gen, verbose=0)
    print(f"[INFO] epoch {epoch+1} | val loss: {val_loss:.4f} | val acc: {val_acc:.4f}")

val_loss, val_acc = model.evaluate(test_gen, verbose=0)
print(f"[INFO] final val loss: {val_loss:.4f} | val acc: {val_acc:.4f}")

model.save("steering_classification_lightcnn.h5")
print("[INFO] saved steering_classification_lightcnn.h5")
