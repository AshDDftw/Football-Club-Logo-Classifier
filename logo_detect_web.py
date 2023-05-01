import numpy as np
import pickle
import streamlit as st
# import pickle
from PIL import Image

# import matplotlib.pyplot as plt
# import seaborn as sns
import cv2
from rembg.bg import remove

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img

# import torch
# from torch import nn
# from torch.nn import CrossEntropyLoss
# from torch.nn.functional import relu
# from torch.optim import Adam
# from torch.utils.data import DataLoader

from torchvision import transforms

# import warnings
# warnings.filterwarnings('ignore')


import tensorflow as tf

# # Save the model
# model.save('model')

# Load the model
loaded_model = tf.saved_model.load('model')

def logo_prediction(input_data):
    test_image = input_data

    classes_reverse_dict={0: 'ac-milan',
    1: 'angers-sco',
    2: 'arsenal',
    3: 'as-monaco',
    4: 'as-saint-etienne',
    5: 'aston-villa',
    6: 'atalanta',
    7: 'athletic',
    8: 'atletico-madrid',
    9: 'augsburg',
    10: 'barcelona',
    11: 'bayern',
    12: 'bologna',
    13: 'bremen',
    14: 'brentford',
    15: 'brighton',
    16: 'burnley',
    17: 'cadiz',
    18: 'cagliari',
    19: 'celta',
    20: 'chelsea',
    21: 'clermont-foot-63',
    22: 'crystal-palace',
    23: 'deportivo-alavez',
    24: 'dortmund',
    25: 'dusseldorf',
    26: 'elche',
    27: 'empoli',
    28: 'espanyol',
    29: 'estac-troyes',
    30: 'everton',
    31: 'fc-girondins-de-bordeaux',
    32: 'fc-lorient',
    33: 'fc-metz',
    34: 'fc-nantes',
    35: 'fiorentina',
    36: 'frankfurt',
    37: 'freiburg',
    38: 'furth',
    39: 'genoa',
    40: 'getafe',
    41: 'granada',
    42: 'hamburg',
    43: 'hannover',
    44: 'hellas-verona',
    45: 'hertha-bsc-berlin',
    46: 'hoffenheim',
    47: 'inter',
    48: 'juventus',
    49: 'lazio',
    50: 'leeds-united',
    51: 'leicester-city',
    52: 'levante',
    53: 'leverkusen',
    54: 'liverpool',
    55: 'losc-lille',
    56: 'mainz',
    57: 'mallorca',
    58: 'manchester-city',
    59: 'manchester-united',
    60: 'moenchengladbach',
    61: 'montpellier-herault',
    62: 'napoli',
    63: 'newcastle-united',
    64: 'norwich-city',
    65: 'nuremberg',
    66: 'ogc-nice',
    67: 'olympique-de-marseille',
    68: 'olympique-lyonnais',
    69: 'osasuna',
    70: 'paris-saint-germain',
    71: 'rayo-vallecano',
    72: 'rc-lens',
    73: 'rc-strasbourg-alsace',
    74: 'real-betis',
    75: 'real-madrid',
    76: 'real-sociedad',
    77: 'redbull-leipzig',
    78: 'roma',
    79: 'salernitana',
    80: 'sampdoria',
    81: 'sassuolo',
    82: 'schalke',
    83: 'sevilla',
    84: 'southampton',
    85: 'spezia',
    86: 'stade-brestois-29',
    87: 'stade-de-reims',
    88: 'stade-rennais-fc',
    89: 'stuttgart',
    90: 'torino',
    91: 'tottenham-hotspur',
    92: 'udinese',
    93: 'valencia',
    94: 'venezia',
    95: 'villarreal',
    96: 'watford',
    97: 'west-ham-united',
    98: 'wolfsburg',
    99: 'wolves'}

        
    # removing image background
    # test_image = remove(test_image)

    IMG_SIZE = 64


    transform = transforms.Compose([transforms.ToTensor()])

   # removing image background
    test_image = remove(test_image)

    # making the transparent background black
    black_background_image = Image.new('RGB', test_image.size, (0,0,0))
    black_background_image.paste(test_image, mask=test_image)

    # resizing the image
    test_image = img_to_array(black_background_image)
    resized_test_image = cv2.resize(test_image, (IMG_SIZE, IMG_SIZE))
    final_test_image = transform(resized_test_image)

    dim1, dim2, dim3 = final_test_image.shape
    final_test_image = final_test_image.reshape(1, dim1, dim2, dim3) 

    final_test_image = final_test_image.to(torch.float32)
    final_test_image = final_test_image.numpy()
    final_test_image = tf.transpose(final_test_image, [0, 2, 3, 1])

    prediction = loaded_model(final_test_image)[0]
    result = classes_reverse_dict[np.argmax(prediction)]

    return result



def main():


    # giving a title
    st.title('Logo Prediction Web App')


    # getting the input data from the user


    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # If the user uploaded a file
    if uploaded_file is not None:
        # Read the contents of the file
        image = Image.open(uploaded_file)

        # Display the image
        st.image(image, caption="Uploaded image", use_column_width=True)


    # code for Prediction
    logo = ''

    # creating a button for Prediction

    if st.button('Logo Test Result'):
        logo = logo_prediction(image)
        
        
    st.success(logo)





if __name__ == '__main__':
    main()
