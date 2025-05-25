import tensorflow as tf

def prepare_data(train_dir='data/training/', test_dir='data/testing/', batch_size=64, img_size=224):
    AUTOTUNE = tf.data.AUTOTUNE

    # Charger dataset d'entraînement
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        label_mode='int',
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True
    )

    # Obtenir les noms de classes immédiatement
    class_names = train_dataset.class_names
    num_classes = len(class_names)

    # Charger dataset de test
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        label_mode='int',
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )

    # Normalisation (0-255 -> [-1, 1])
    normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)

    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, test_dataset, class_names
