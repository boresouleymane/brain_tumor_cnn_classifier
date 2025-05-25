import tensorflow as tf
from tensorflow.keras import layers, models, applications

def get_pretrained_model(num_classes=4, img_size=224):
    base_model = applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Geler les couches convolutives

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = tf.image.resize(x_train, (224, 224))
    x_test = tf.image.resize(x_test, (224, 224))

    model = get_pretrained_model(num_classes=10)

    model.fit(x_train, y_train, epochs=3, validation_split=0.1)

    # Sauvegarde au format Keras 3 (.keras)
    model.save("souleymane_bore_model.keras")
    print("✅ Modèle enregistré au format .keras")
