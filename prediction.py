from create_Segmentation_model import my_Unet
from important_variables import img_height, img_width
import streamlit as st
from PIL import Image
import numpy as np



def segment_image(model_path):
    model = my_Unet(model_path)

    input_col, output_col = st.columns([1, 1])

    input_col.markdown('## Input Image')
    output_col.markdown('## Segmented Image')
    uploaded_file = input_col.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        # image = cv2.imread(uploaded_file)
        image = Image.open(uploaded_file)
        input_col.image(image, caption='Input Image.', use_column_width=True)
        st.write("")
        image = image.resize((img_width, img_height))
        
        image = np.expand_dims(image, 0)/255

        
        pred = np.argmax(model.predict(image), axis=3)[0,:,:]
        

        print(pred.shape)

        output_col.image(pred, caption='Segmented Image.', use_column_width=True)

        
        