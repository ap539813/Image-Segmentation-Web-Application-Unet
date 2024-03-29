import streamlit as st
from important_variables import theme_image_name, model_path_unet, model_path_encoder_decoder
from prediction import segment_image

def main():
    # data = generate_data()
    st.sidebar.image(theme_image_name)

    st.sidebar.title("Control Panel")

    type_model = st.sidebar.radio("Select Type of Model: ", ('Unet Model', 'Simple Encoder Decoder'))

    if type_model == 'Unet Model':
        st.title(f"Segmentation using {type_model}")

        segment_image(model_path_unet)

    elif type_model == 'Simple Encoder Decoder':
        st.title(f"Segmentation using {type_model}")

        segment_image(model_path_encoder_decoder)

    

def homepage():
    # if st.session_state['home_page']:
    home_image = st.image(theme_image_name)

#     markdown_image = st.markdown(
#     f'<img style="max-width: 100%; height: auto;" src="data:image/gif;base64,{data_url}" alt="homepage gif">',
#     unsafe_allow_html=True,
# )

    c1, c2, c3 = st.columns([2,1,2])

    # c2.markdown('<div style="text-align: center;"> <h2> Developer Arnav Singhal </h2></div>', unsafe_allow_html=True)

    c2.markdown('')
    c2.markdown('')
    continue_forward = c2.button('Continue >>>')

    st.session_state['home_page'] = False
    # print(search_page)
    

    if continue_forward:
        print('going to the application!!')
        home_image.empty()
        main()


