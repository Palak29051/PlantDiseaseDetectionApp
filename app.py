import streamlit as st
import cv2
import numpy as np
from joblib import load
import mahotas
from PIL import Image

# Converting each image to RGB from BGR format
def rgb_bgr(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img

# Conversion to HSV image format from RGB
def bgr_hsv(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img

# image segmentation for extraction of green and brown color
def img_segmentation(rgb_img,hsv_img):
    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img,rgb_img, mask=healthy_mask)
    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bins=8
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Streamlit app
def main():

    st.set_page_config(page_title="Plant Disease Detection App",page_icon="🌵")

    st.sidebar.title("Plant Disease Detection App")

    image = Image.open('header_image.png')
    st.sidebar.image(image)

    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, caption='Uploaded Image')

        fixed_size = (500, 500)
        test_image = cv2.resize(image, fixed_size)
        
        # Perform image processing and prediction
        RGB_BGR = rgb_bgr(test_image)
        BGR_HSV = bgr_hsv(RGB_BGR)
        IMG_SEGMENT = img_segmentation(RGB_BGR, BGR_HSV)
        
        fv_hu_moments = fd_hu_moments(IMG_SEGMENT)
        fv_haralick = fd_haralick(IMG_SEGMENT)
        fv_histogram = fd_histogram(IMG_SEGMENT)
        
        test_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        
        scaler = load('output/minmaxscaler.joblib')
        test_feature = scaler.transform(test_feature.reshape(1, -1))

        model = load('output/random_forest_model.joblib')
        prediction = model.predict(test_feature)

        encoder = load('output/labelencoder.joblib')
        prediction = encoder.inverse_transform(prediction)

        st.subheader("Prediction")
        st.write(prediction[0])

if __name__ == "__main__":
    main()