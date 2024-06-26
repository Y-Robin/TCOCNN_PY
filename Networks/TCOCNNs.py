import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
from skopt import gp_minimize
from skopt.space import Integer, Real
import json
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
import copy
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
from scipy.ndimage import uniform_filter1d


class TCOCNNClass:
    def __init__(self, input_size, output_size, regression=True, optim_params=None):
        self.input_size = input_size
        self.output_size = output_size
        self.regression = regression
        self.model = None
        self.built_flag = False
        self.optim_params = optim_params
        self.history = None
        self.l2_reg =  0.0001
        self.trainFlag = False

        if optim_params is not None:
            self.build_net(optim_params)

    def build_net(self, optim):
        self.built_flag = True
        n_filters = optim['n_filter']
        first_kernel_size = optim['kernel']
        first_stride = optim['stride']
        dropout_rate = optim['drop_out']
        num_convs = optim['section_depth']
        width_fc = optim['num_neurons']

        self.optim_params = optim

        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=self.input_size))
        
        # First Convolutional Block
        model.add(layers.Conv2D(n_filters, (1, first_kernel_size), strides=(1, 1), padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(1, first_stride)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Conv2D(n_filters, (1, first_kernel_size), strides=(1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        # Additional Convolutional Blocks
        for i in range(1, num_convs):
            model.add(layers.Conv2D(n_filters * (i + 1), (1, 2), strides=(1, 1), padding='same'))
            model.add(layers.MaxPooling2D(pool_size=(1, 2)))
            model.add(layers.BatchNormalization())
            model.add(layers.ReLU())
            model.add(layers.Conv2D(n_filters * (i + 1), (1, 2), strides=(1, 1), padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.ReLU())

        model.add(layers.GlobalMaxPooling2D())
        # Fully Connected Layers
        model.add(layers.Flatten())
        model.add(layers.Dense(width_fc, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(self.output_size))

        if not self.regression:
            model.add(layers.Softmax())

        self.model = model

    def compile_model(self, initial_learning_rate=1e-3):

        def scheduler(epoch, lr):
            if epoch > 0 and epoch % 2 == 0:
                return lr * 0.9
            return lr

        lr_scheduler = callbacks.LearningRateScheduler(scheduler)

        optimizer = optimizers.Adam(learning_rate=initial_learning_rate)
        if self.regression:
            self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        else:
            self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        self.lr_scheduler = lr_scheduler

        # Print the model summary
        # self.model.summary()

    def train(self, data, target, validation_data=None, epochs=75, batch_size=50):
        if self.built_flag:
            self.history = self.model.fit(
                data, target, epochs=epochs, batch_size=batch_size, callbacks=[self.lr_scheduler],verbose = 0
            )

            # Get the list of layers; Population statisicts
            layersAll = self.model.layers
            for i, layer in enumerate(layersAll):
                if isinstance(layer, layers.BatchNormalization):
                    prev_layer = layersAll[i - 1]
                    intermediate_layer_model = tf.keras.Model(inputs=self.model.input,
                                          outputs=prev_layer.output)
                    pred = intermediate_layer_model.predict(data,batch_size = batch_size)
                    layer.moving_mean.assign(np.mean(pred,axis=(0, 1, 2)))
                    layer.moving_variance.assign(np.var(pred,axis=(0, 1, 2)))


            self.trainFlag = True 
            # Recalculate batch normalization statistics
            # self.recalculate_batchnorm_statistics(data)
        else:
            print("Model Not Built")

    def optimize_model(self, data, target, validation_data, validation_target, num_epochs=50):

        def objective(params):
            self.build_net({
                'n_filter': params[0],
                'section_depth': params[1],
                'kernel': params[2],
                'stride': params[3],
                'num_neurons': params[4],
                'drop_out': params[5],
            })
            self.compile_model(params[6])
            try:
                self.train(data, target)
                val_loss = self.model.evaluate(validation_data, validation_target, verbose=0)
                tf.keras.backend.clear_session()
                # Clear the memory
                del self.model
                gc.collect() 
                
                return val_loss[0]
            except tf.errors.ResourceExhaustedError as e:
                print("Error")
                tf.keras.backend.clear_session()
                return float('inf')

        space = [
            Integer(50, 150, name='n_filter'),
            Integer(3, 5, name='section_depth'),
            Integer(5, 15, name='kernel'),
            Integer(2, 5, name='stride'),
            Integer(800, 1500, name='num_neurons'),
            Real(0.1, 0.5, name='drop_out'),
            Real(1e-5, 1e-3, prior='log-uniform', name='initial_learning_rate')
        ]

        res = gp_minimize(objective, space, n_calls=num_epochs, random_state=42)
        best_params = res.x
        self.build_net({
            'n_filter': best_params[0],
            'section_depth': best_params[1],
            'kernel': best_params[2],
            'stride': best_params[3],
            'num_neurons': best_params[4],
            'drop_out': best_params[5],
        })
        self.compile_model(best_params[6])
        self.train(data, target)


        # Save the best parameters to a JSON file
        best_params_dict = {
            'n_filter': int(best_params[0]),
            'section_depth': int(best_params[1]),
            'kernel': int(best_params[2]),
            'stride': int(best_params[3]),
            'num_neurons': int(best_params[4]),
            'drop_out': float(best_params[5]),
            'initial_learning_rate': float(best_params[6])
        }


        with open("../Evaluation/test.json", 'w') as f:
            json.dump(best_params_dict, f)


        # Plot validation errors from res
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(res.func_vals) + 1), res.func_vals, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Validation Error')
        plt.title('Validation Errors Over Iterations')
        plt.grid(True)
        plt.show()

        return res
    
    def custom_occlusion(self, data_train, data_test, method='custom'):
        # Method for explainable AI
        # Input of training data is done to get a reference sample
        # Testdata is the data that is analyzed

        activation_map = np.zeros(data_test.shape)
        mean_train_sample = np.mean(data_train, axis=0)


        kernel_half = 15
        stride = 10

        for i in range(data_test.shape[0]):
            data_occ = data_test[i, ...]
            a_org = self.model.predict(np.expand_dims(data_occ, axis=0))
            z = 1
            data_occ_all = np.zeros((int(np.ceil(data_train.shape[2] / stride) * 4),data_train.shape[1], data_train.shape[2], 1))

            for j in range(data_train.shape[1]):
                x = []
                for k in range(int(np.ceil(data_train.shape[2] / stride))):
                    data_occ = data_test[i,...].copy()
                    if method == 'subsensor':
                        data_occ[j, max(0, (k) * stride - kernel_half):min(data_train.shape[2], (k) * stride + kernel_half + 1)] = np.mean(data_test[j, :, :, i])
                    elif method == 'custom':
                        data_occ[j, max(0, (k) * stride - kernel_half):min(data_train.shape[2], (k) * stride + kernel_half + 1)] = mean_train_sample[j, max(0, (k) * stride - kernel_half):min(data_train.shape[2], (k) * stride + kernel_half + 1)]
                    else:
                        data_occ[j, max(0, (k) * stride - kernel_half):min(data_train.shape[2], (k) * stride + kernel_half + 1)] = np.mean(data_test[:, :, :, i])

                    data_occ_all[z - 1,:, :, 0] = data_occ[...,0]
                    x.append(min(max(0, (k) * stride), data_train.shape[2]))
                    z += 1
            pred = self.model.predict(data_occ_all)
            a_single_one = (np.abs(self.zscore(pred - a_org)) / (a_org + 1)).squeeze()
            a_single = np.zeros((data_train.shape[1], int(np.ceil(data_train.shape[2] / stride))))
            for lv1 in range(data_train.shape[1]):
                a_single[lv1, :] = a_single_one[(int(np.ceil(data_train.shape[2] / stride)) * lv1):(int(np.ceil(data_train.shape[2] / stride)) * (lv1 + 1))]
            
            
            #Upsampling
            temp = np.zeros(data_train.shape)
            a_single_up = temp[0,...]
            x = np.linspace(1, data_train.shape[2], int(data_train.shape[2] / stride))
            x2 = np.arange(1, data_train.shape[2] + 1)
            
            for lv1 in range(data_train.shape[1]):
                a_single_up[lv1, :, 0] = np.interp(x2, x, a_single[lv1, :])

            activation_map[i, :, :, 0] = a_single_up[:, :, 0]

        return activation_map

    def zscore(self,a, axis=0, ddof=0):
        mean = np.mean(a, axis=axis)
        std = np.std(a, axis=axis, ddof=ddof)
        return (a - mean) / std

    def get_gradient_map(self, data, layer_name=None, window_size=5):
        
        # Ensure data is a tensor
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(data)
            # Forward pass
            predictions = self.model(data)
            
            # If layer_name is specified, get the output of that layer
            if layer_name:
                layer_output = self.model.get_layer(layer_name).output
                predictions = layer_output

            # Compute the loss (mean of predictions in this case)
            loss = tf.reduce_mean(predictions)
        
        # Compute gradients of the loss with respect to the input data
        gradients = tape.gradient(loss, data)
        
        # Convert gradients to numpy array
        gradients = gradients.numpy()
        
        # Z-score normalization for each sample and subsensor independently
        gradients = np.abs(gradients)
        
        # Apply sliding window smoothing along the second dimension
        smoothed_gradients = np.zeros_like(gradients)
        for i in range(gradients.shape[0]):  # iterate over samples
            for j in range(gradients.shape[1]):  # iterate over subsensors
                smoothed_gradients[i, j, :, 0] = uniform_filter1d(gradients[i, j, :, 0], size=window_size)
        
        return smoothed_gradients

    def retrain(self, new_data, new_target, validation_data=None, epochs=75, batch_size=50, fine_tune=False, new_learning_rate=None):
        if not self.built_flag:
            raise RuntimeError("Model is not built. Please build the model before retraining.")

        if fine_tune:
            # Fine-tune specific layers (e.g., last few layers)
            for layer in self.model.layers[:-2]:
                layer.trainable = False
            for layer in self.model.layers[-2:]:
                layer.trainable = True


        if new_learning_rate is not None:
            # Update the optimizer with the new learning rate
            optimizer = optimizers.Adam(learning_rate=new_learning_rate)
            if self.regression:
                self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            else:
                self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Retrain the model
        self.history = self.model.fit(
            new_data, new_target, validation_data=validation_data, epochs=epochs, batch_size=batch_size,
            callbacks=[self.lr_scheduler],verbose = 0
        )

        # Get the list of layers; Population statisicts
        layersAll = self.model.layers
        for i, layer in enumerate(layersAll):
            if isinstance(layer, layers.BatchNormalization):
                prev_layer = layersAll[i - 1]
                intermediate_layer_model = tf.keras.Model(inputs=self.model.input,
                                        outputs=prev_layer.output)
                pred = intermediate_layer_model.predict(new_data,batch_size = batch_size)
                layer.moving_mean.assign(np.mean(pred,axis=(0, 1, 2)))
                layer.moving_variance.assign(np.var(pred,axis=(0, 1, 2)))

        print("Retraining completed.")

    def predict(self, data):
        return self.model.predict(data)
    
    def copy(self):
        # Create a new instance of TCOCNN with the same initialization parameters
        new_instance = self.__class__(self.input_size, self.output_size, self.regression, self.optim_params)
        
        if self.model is not None:
            # Save model architecture to JSON
            model_json = self.model.to_json()
            new_instance.model = model_from_json(model_json)
            
            # Copy model weights
            new_instance.model.set_weights(copy.deepcopy(self.model.get_weights()))
            
            # Copy additional attributes
            new_instance.built_flag = self.built_flag
            #new_instance.history = copy.deepcopy(self.history)
            new_instance.lr_scheduler = self.lr_scheduler
            
            return new_instance

# Example usage:
# input_size = (64, 64, 3)  # Example input size
# output_size = 10  # Example output size for classification
# model = TCOCNN(input_size, output_size, regression=False)
# model.build_net({'n_filter': 64, 'section_depth': 3, 'kernel': 3, 'stride': 2, 'num_neurons': 128, 'drop_out': 0.3})
# model.compile_model(initial_learning_rate=1e-3)
# # Assume data_train, target_train, data_val, target_val are prepared
# model.train(data_train, target_train, validation_data=(data_val, target_val), epochs=10, batch_size=32)
# predictions = model.predict(data_test)

# Example usage:
# input_size = (64, 64, 3)  # Example input size
# output_size = 10  # Example output size for classification
# model = TCOCNN(input_size, output_size, regression=False)
# model.build_net({'n_filter': 64, 'section_depth': 3, 'kernel': 3, 'stride': 2, 'num_neurons': 128, 'drop_out': 0.3})
# model.compile_model(initial_learning_rate=1e-3)
# # Assume data_train, target_train, data_val, target_val are prepared
# model.train(data_train, target_train, validation_data=(data_val, target_val), epochs=10, batch_size=32)
# predictions = model.predict(data_test)
