from keras import backend as K
import base64

# Define the hight and width of the images as required

img_width, img_height = 256,256

input_shape = (img_width, img_height)

# dimensions of our images.
# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_height)
# else:
#     input_shape = (img_width, img_height, 3)


theme_image_name = 'Plant disease classification.png'

"""### gif from local file"""
file_ = open(theme_image_name, "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()


css_file_path = 'style/style.css'

model_path_unet = 'Unet_model.hdf5'
model_path_encoder_decoder = 'Simple_Encode_Decoder.hdf5'

