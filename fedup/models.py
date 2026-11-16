"""Models used in the experiments: the CNNs of the label-flipping settings and
ResNet-18 for the backdoor setting."""

import tensorflow as tf


def _bn_relu_conv(filters, kernel_size, **kwargs):
    return [tf.keras.layers.Conv2D(filters, kernel_size, padding="same", **kwargs),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()]


def cifar10_cnn(num_classes=10):
    layers = [tf.keras.Input(shape=(32, 32, 3))]
    for filters in (32, 64, 128):
        layers += _bn_relu_conv(filters, 3) + _bn_relu_conv(filters, 3)
        layers.append(tf.keras.layers.MaxPooling2D(2))
    layers += [tf.keras.layers.Flatten(),
               tf.keras.layers.Dense(128), tf.keras.layers.ReLU(),
               tf.keras.layers.Dense(num_classes, activation="softmax")]
    model = tf.keras.Sequential(layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fashion_mnist_cnn(num_classes=10):
    layers = [tf.keras.Input(shape=(28, 28, 1))]
    for filters in (16, 32):
        layers += _bn_relu_conv(filters, 5)
        layers.append(tf.keras.layers.MaxPooling2D(2))
    layers += [tf.keras.layers.Flatten(),
               tf.keras.layers.Dense(num_classes, activation="softmax")]
    model = tf.keras.Sequential(layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, strides=1, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, 1, strides=stride),
                tf.keras.layers.BatchNormalization()])
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        residual = self.downsample(inputs)
        x = tf.nn.relu(self.bn1(self.conv1(inputs), training=training))
        x = self.bn2(self.conv2(x), training=training)
        return tf.nn.relu(x + residual)


class ResNet18(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")
        self.blocks = [tf.keras.Sequential([BasicBlock(f, stride), BasicBlock(f)])
                       for f, stride in ((64, 1), (128, 2), (256, 2), (512, 2))]
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=None):
        x = self.pool1(tf.nn.relu(self.bn1(self.conv1(inputs), training=training)))
        for block in self.blocks:
            x = block(x, training=training)
        return self.fc(self.avgpool(x))


def resnet18(num_classes=10):
    model = ResNet18(num_classes)
    model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model(tf.zeros((1, 32, 32, 3)))
    return model


MODELS = {"cifar10_cnn": cifar10_cnn, "fashion_mnist_cnn": fashion_mnist_cnn, "resnet18": resnet18}
