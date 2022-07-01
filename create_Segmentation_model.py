# from keras.models import Model
# from keras.layers import Conv2D, MaxPooling2D, MaxPool2D, Conv2DTranspose
# from keras.layers import Dropout, Input, concatenate

import keras.models as models
from utils import jacard_loss
# from keras.regularizers import l2
def my_Unet(model_path):
    model = models.load_model(model_path, custom_objects = {'jacard': jacard_loss})
    return model

# def my_Unet(n_classes=2, channels=3, input_shape = (256, 256)):
#     width, height = input_shape
#     inputs = Input((height, width, channels))

#     # Encoding Path
#     conv_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
#     conv_1 = Dropout(0.1)(conv_1)  
#     conv_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_1)
#     pool_1 = MaxPooling2D((2, 2))(conv_1)
    
#     conv_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_1)
#     conv_2 = Dropout(0.1)(conv_2)  
#     conv_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_2)
#     pool_2 = MaxPooling2D((2, 2))(conv_2)
     
#     conv_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_2)
#     conv_3 = Dropout(0.1)(conv_3)
#     conv_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_3)
#     pool_3 = MaxPooling2D((2, 2))(conv_3)
     
#     conv_4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_3)
#     conv_4 = Dropout(0.1)(conv_4)
#     conv_4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_4)
#     pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)
     
#     conv_5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_4)
#     conv_5 = Dropout(0.2)(conv_5)
#     conv_5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_5)
    
#     # Decoding Path
#     u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv_5)
#     u6 = concatenate([u6, conv_4])
#     conv_6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
#     conv_6 = Dropout(0.2)(conv_6)
#     conv_6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_6)
     
#     u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv_6)
#     u7 = concatenate([u7, conv_3])
#     conv_7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
#     conv_7 = Dropout(0.1)(conv_7)
#     conv_7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_7)
     
#     u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv_7)
#     u8 = concatenate([u8, conv_2])
#     conv_8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
#     conv_8 = Dropout(0.2)(conv_8)  
#     conv_8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_8)
     
#     u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv_8)
#     u9 = concatenate([u9, conv_1], axis=3)
#     conv_9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
#     conv_9 = Dropout(0.1)(conv_9)
#     conv_9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_9)
     
#     outputs = Conv2D(n_classes, (1, 1), activation='softmax')(conv_9)
    
#     model = Model(inputs=[inputs], outputs=[outputs])
#     return model



