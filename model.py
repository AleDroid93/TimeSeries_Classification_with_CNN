import tensorflow as tf

class MyCnn(tf.keras.Model):
    def __init__(self, filters, n_classes, kernel_size, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='convNetwork',
                 **kwargs):
        # chiamata al costruttore della classe padre, Model
        super(MyCnn, self).__init__(name=name, **kwargs)
        # definizione dei layers del modello
        self.conv1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation=hidden_activation)
        self.batchNorm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.conv2 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation=hidden_activation)
        self.batchNorm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dens1 = tf.keras.layers.Dense(units=filters*2, activation=hidden_activation)
        self.dens2 = tf.keras.layers.Dense(units=filters*2, activation=hidden_activation)
        self.model_output = tf.keras.layers.Dense(units=n_classes, activation=output_activation)

    def call(self, inputs, training=False):
        # definisco il flusso, che la rete rappresentata dal modello, deve seguire.
        inputs = self.conv1(inputs)
        inputs = self.batchNorm1(inputs, training=training)
        inputs = self.dropout1(inputs, training=training)
        inputs = self.conv2(inputs)
        inputs = self.batchNorm2(inputs, training=training)
        inputs = self.dropout2(inputs, training=training)
        inputs = self.flatten_layer(inputs)
        inputs = self.dens1(inputs)
        inputs = self.dens2(inputs)
        return self.model_output(inputs)