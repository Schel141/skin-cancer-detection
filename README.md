# Skin Cancer Detection

This project focuses on the detection of skin cancer based on dermatoscopic images using Convolutional Neural Networks (CNN). We built our own CNN to classify the dermatoscopic images into seven categories (please see Data Source for a description of the categories). In addition, we used different pretrained models (e.g. VGG16, ResNet-50, MobileNet V2). The best results were achieved with the ResNet-50 model (F1-score: 0.79). Furthermore, we created a website (https://skicadetec.herokuapp.com/) with streamlit, to detect the class of a skin lesion image with our model. This can contribute to the early detection of skin cancer. The code for the website can be found at https://github.com/Schel141/SkicadeApp.

# Data Source

We used the HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. It consists of 10015 dermatoscopic images. Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).

Link to the dataset: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000


