import altair as alt
import streamlit as st
import seaborn as sns

# importing the local modules
from important_variables import input_shape, data_url, css_file_path, theme_image_name
# from prediction import prediction_individual_plant, prediction_general
from homepage import main, homepage
from add_style import local_css
import tensorflow as tf

from PIL import Image


## Basic setup and app layout
st.set_page_config(layout="wide")

alt.renderers.set_embed_options(scaleFactor=2)


local_css(css_file_path)

print(st.session_state)

if 'home_page' not in st.session_state:
    st.session_state['home_page'] = True




    


if __name__ == '__main__':
    if st.session_state['home_page']:
        homepage()
    else:
        main()

