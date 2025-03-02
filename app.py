import streamlit as st
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
# model prediction

def prediction(pred_image):
  model = load_model('Chilli_MobileNet_model.h5')  # Ensure correct loading

  if pred_image is None:
    return None  # Handle case where no image is uploaded

  # Resize the image correctly to (224, 224)
  image = load_img(pred_image, target_size=(224, 224))  # ✅ Fixed to 224x224
  image_array = img_to_array(image)  # Convert to NumPy array
  image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
  # Normalize image (if your model expects it)
  image_array = image_array / 255.0  # ✅ Ensure values are between 0 and 1
  # Model prediction
  prediction = model.predict(image_array)
  result_index = np.argmax(prediction)
  return result_index

st.sidebar.title('Dashboard')
app_page_select = st.sidebar.selectbox("Select Page ",['Home','App','About','Contact']) 
st.sidebar.markdown("""
                    
- Design and Developed By Mahaveer
- Under guidance of Dr. Sachidanand Chaurasia and Dr. Srinivasa N
- Developed using the Deep Learning CNN and Transfer Learning


""")
if(app_page_select =="Home"):
  image_path ="logo.png"
  st.image(image_path,use_container_width =True)
  st.header("Chilli Infestation Detection System")
  image_path ="classes.png"
  st.image(image_path,use_container_width =True)
  st.markdown("""
      ## 🌾 Early Chilli Infestation Detection Using AI
      ### Protect Your Crops, Maximize Your Yield

      Chilli  farming is the backbone of global food security, but crop Infestations can lead to devastating losses. At Our Platform, we empower farmers with AI-driven technology to detect Chilli Infestations early—helping you take action before it’s too late.


      ### AI-Powered Chilli Infestation Detection
      Using advanced Deep Learning (CNN - MobileNet), our platform accurately identifies four major Chilli Infestations:
      
      - ✅ Affected - Identify the chilli Infestation
      - ✅ Healthy - Identify the Healthy Images
      - ✅ **Instant Infestation Detection** – Upload a leaf image and get results in seconds.  
      - ✅ **Cloud-Based AI Model** – Powered by CNN and transfer learning for high accuracy.  
      - ✅ **Comprehensive Infestation Coverage** 
      - ✅ **Accessible Anytime, Anywhere** – Works seamlessly on mobile and desktop.  

      - 🔍 **Start diagnosing now! Click below to use our web app.**  
    ### How It Works
    - 📸 Upload an Image – Take a photo of the affected Chilli  plant.
    - 🤖 AI Analysis – Our deep learning model detects the Infestation.
    - 📊 Get Instant Results – Receive Infestation insights and prevention tips.
        
     ***Start Protecting Your Chilli  Crops Today!
      Take control of your harvest with smart technology. Detect Chilli Infestations early and safeguard your yield.***
  """)

elif(app_page_select=="App"):
  st.header("App - Page")
  image_path ="classes.png"
  st.image(image_path,use_container_width =True)
  test_image =st.file_uploader("Choose and image - up to 200MB")
  if(st.button("Show Image")):
    st.image(test_image,use_container_width=True)
  if(st.button("Predict the image")):

    st.spinner()
    time.sleep(2)
    classes =['Affected','Healthy']
    result_index = prediction(test_image)
    st.success("Model  is predicted {}".format(classes[result_index]))


elif(app_page_select=="About"):
  st.header("About - Page")
 
  st.markdown("""
  ----------------------------------
  ### 🌾 Empowering Farmers with Early Chilli Infestation Detection

    Welcome to out platform, where technology meets agriculture to protect one of the world’s most essential crops—Chilli . We are dedicated to helping farmers detect and manage Chilli Infestations early using cutting-edge deep learning techniques.

  ### Our Mission
    Chilli  is a staple food for millions, and its health is crucial for food security and farmer livelihoods. Our mission is to provide farmers with an intelligent, easy-to-use tool for early detection of Chilli Infestations, ensuring timely intervention and improved crop yield.

  ### How It Works
    Our platform leverages MobileNet, a deep learning-based Convolutional Neural Network (CNN), to accurately identify four major Chilli Infestations:

    - Affected
    - Healthy

    By simply uploading an image of the affected plant, our model analyzes and classifies the Infestation, providing farmers with real-time insights for effective Infestation management.
             
  ### 🔬 Technology Behind the Platform:  
  ------------------------------------------
    - **Deep Learning Models** – We use **CNNs and Transfer Learning** for high-precision detection.  
    - **Cloud Deployment** – Our AI model runs in the cloud for **real-time predictions**.  
    - **User-Friendly Interface** – A seamless web app designed for ease of use.  

   ##### 👨‍💻 Developed By: This project is built by **Mahaveer**, a passionate AI researcher specializing in **deep learning applications domain**. 
    

  """)
  image_path_graph ="classes.png"
  st.image(image_path_graph,use_container_width =True)
  image_path_matrix ="confusionmatrix.png"
  st.image(image_path_matrix,use_container_width =True)

elif(app_page_select=="Contact"):
  st.header("Contact US - Page")
  st.markdown("""
    --------------------------------------------
    We’d love to hear from you! Whether you have a question, need assistance, or just want to say hello, feel free to reach out to us.

    
    ### Get in Touch  
    -----------------------
      - 📍 Address: [Your Business Address]
      - 📞 Phone: [Your Contact Number]
      - 📧 Email: [Your Email Address]
    
    ### Business Hours
    --------------------
      - 🕒 Monday – Friday: 9:00 AM – 6:00 PM
      - 🕒 Saturday: 10:00 AM – 4:00 PM
      - ❌ Sunday: Closed
              
    ### Follow Us
    --------------------
        Stay connected and follow us on social media for updates, news, and special offers.

        🔗 [Facebook] | [Instagram] | [Twitter] | [LinkedIn]

        We look forward to assisting you!
  """)
