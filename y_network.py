#y_network
img_size=64
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, concatenate

#ynetwork function
def y_network(x_train, x_test, y_train, y_test, epochs, layer, optimizer, dropout):
    tf.keras.backend.clear_session()

    input_shape= (img_size, img_size, 3)
    n_filters=32
    batch_size=8
    
    #left branch
    left_input = Input(shape=input_shape)
    xl=left_input
    filters=n_filters

    for l in range(layer):
        xl=Conv2D(filters=filters, kernel_size= (3,3), padding='same', activation='relu', dilation_rate=1)(xl)
        xl=Dropout(dropout)(xl)
        xl=MaxPooling2D(pool_size=(2,2))(xl)
        filters*=2

    #right branch
    right_input=Input(shape=input_shape)
    xr=right_input
    filters=n_filters

    for l in range(layer):
        xr=Conv2D(filters=filters, kernel_size = (3,3), padding='same', activation='relu', dilation_rate=2)(xr)
        xr=Dropout(dropout)(xr)
        xr=MaxPooling2D(pool_size=(2,2))(xr)
        filters*=2

    #perceptron
    x=concatenate([xl, xr])
    x=Flatten()(x)
    x=Dropout(dropout)(x)
    output=Dense(1, activation='linear')(x)

    model=Model([left_input, right_input], output)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    history=model.fit([x_train, x_train], y_train, epochs=epochs, validation_data =([x_test, x_test], y_test), batch_size=batch_size)
    test_loss, test_acc=model.evaluate([x_test, x_test], y_test, batch_size=batch_size, verbose=0)
    train_loss, train_acc=model.evaluate([x_train, x_train], y_train, batch_size=batch_size, verbose=0)
    
    return model, test_acc, history

# for layer in layer_list:
#     for optimizer in optimizer_list:
#         for dropout in dropout_list:
#             tf.keras.backend.clear_session()
#             model, accuracy, history = ynetwork(x_train, x_test, y_train, y_test, 30, layer, optimizer, dropout)
#             model_list.append([accuracy, layer, optimizer, dropout])

#             plt.plot(history.history['accuracy'])
#             plt.plot(history.history['val_accuracy'])
#             plt.title('model accuracy')
#             plt.ylabel('accuracy'), plt.xlabel('epoch')
#             plt.legend(['train', 'test'], loc='upper left')
#             plt.show()
            
#             plt.plot(history.history['loss'])
#             plt.plot(history.history['val_loss'])
#             plt.title('model loss')
#             plt.ylabel('loss'), plt.xlabel('epoch')
#             plt.legend(['train', 'test'], loc='upper left')
#             plt.show()

# #constants
# batch_size=8
# n_filters=32
# label_categories = 2
# input_shape= (img_size, img_size, 3)

# #experiments
# layer_list=[2,3,4]
# optimizer_list=['SGD','adam']
# dropout_list=[0, 0.333]
# model_list=[]

# model, accuracy, history = ynetwork(x_train, x_test, y_train, y_test, 30, 1, 'adam', 0.35)
# y_pred = model.predict([X_test, X_test])
# results = pd.DataFrame({
#     'True Regression Value': y_test.flatten(),  # Flatten y_test if it's a multi-dimensional array
#     'Predicted Regression Value': y_pred.flatten()  # Flatten y_pred for consistency
# })
# results['Percentage Difference'] = (abs(results['True Regression Value'] - results['Predicted Regression Value']) / results['True Regression Value']) * 100
