import cv2
import streamlit as st
import mediapipe as mp
from PIL import Image
import numpy as np
import face_recognition

# Drawing utility
mp_drawing = mp.solutions.drawing_utils
# Face detection utility
mp_face_detection = mp.solutions.face_detection
#model for face detecting the face
model_detection = mp_face_detection.FaceDetection()
#selfie detection utility
mp_selfie_segmentation = mp.solutions.selfie_segmentation
#mosel for selfie segmentation
model= mp_selfie_segmentation.SelfieSegmentation(model_selection = 1)
bg_image = None

drawing_spec = mp_drawing.DrawingSpec((255, 220, 0), thickness=2, circle_radius=1)



#main code begins
st.title("This application performs various operations on images")
st.subheader("Face Recongnition, Face Detection, Selfie segmentation, Image blending")

add_selectbox = st.sidebar.selectbox(
    "select operation from below box",
    ("----","Face Recognition","Face detection", "Selfie segmentation", "Image blending"))

image = st.sidebar.file_uploader("upload a image")

if image is not None:
    image = Image.open(image)
    image = np.array(image)


    
else:
    add_selectbox = "----"



if add_selectbox == "Face detection":
    
    st.sidebar.image(image)
    results = model_detection.process(image)
    
    for landmark in results.detections:
        mp_drawing.draw_detection(image,landmark)

        st.image(image)

elif add_selectbox == "Selfie segmentation":
    colour = st.sidebar.radio("choose background colour for your image",
     ('blue', 'green', "red",'black',"white"))
    st.sidebar.image(image)
    if colour == "blue":
        BG_COLOR = (0,0, 255)
    elif colour == "green":
        BG_COLOR = (0,255, 0)
    elif colour == "black":
        BG_COLOR = (0,0,0)
    elif colour == "red":
        BG_COLOR = (255,0,0)
    elif colour == "white":
        BG_COLOR = (255,255, 255)

    result = model.process(image)
    condition = np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.35
    if bg_image is None:

        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
    output_image = np.where(condition, image, bg_image)
    st.image(output_image)

elif add_selectbox == "Image blending":  
    image1 = st.sidebar.file_uploader("upload a image for blending")
    if image1 is not None: 
        image1 = Image.open(image1)
        image1 = np.array(image1)    
    

        if image.shape[0] == image1.shape[0] and image.shape[1] == image1.shape[0]:
            blended_image = cv2.addWeighted(image, 0.6, image1, 0.4, gamma=0.2)
            st.image(blended_image)
        elif image.shape[0] > image1.shape[0] and image.shape[1] > image1.shape[0]:
            image1 = cv2.resize(image1,(image.shape[1],image.shape[0]))
            blended_image = cv2.addWeighted(image, 0.6, image1, 0.4, gamma=0.2)
        
            st.image(blended_image)
        elif image.shape[0] < image1.shape[0] and image.shape[1] < image1.shape[0]:
            image = cv2.resize(image,(image1.shape[1],image1.shape[0]))
            blended_image = cv2.addWeighted(image, 0.6, image1, 0.4, gamma=0.2)
        
            st.image(blended_image)
        else:
            st.write("upload the corrrect size of images")


    else:
        st.write("Upload a image")

elif add_selectbox == "Face Recognition":

    image1 = st.sidebar.file_uploader("upload a image for recognition")
    if image1 is not None: 
        image1 = Image.open(image1)
        image1 = np.array(image1)

        #image_train = cv2.imread(image)

        #image_train = cv2.resize(image,None,fx = 0.5,fy = 0.4)
#image_train = face_recognition.load_image_file("images\\chris.jpg")

        image_location_train = face_recognition.face_locations(image)[0]
        image_encoding_train = face_recognition.face_encodings(image)[0]



        #image_test = face_recognition.load_image_file(image1)
        image_location_test = face_recognition.face_locations(image1)[0]
        image_encoding_test = face_recognition.face_encodings(image1)[0]
        

        



        result = face_recognition.compare_faces([image_encoding_test],image_encoding_train)
        dst = face_recognition.face_distance([image_encoding_train],image_encoding_test)



        if result == True:
            st.subheader(result)

            image_train = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
            cv2.rectangle(image_train, (image_location_train[3],image_location_train[0]),
                               (image_location_train[1],image_location_train[2]), (0,255,0),2)

            cv2.putText(image_train,f"{result} {dst}",(60,60),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255))

            image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
    
            cv2.rectangle(image1, (image_location_test[3],image_location_test[0]),
                               (image_location_test[1],image_location_test[2]), (0,255,0),2)

            cv2.putText(image1,f"{result} {dst}",(60,60),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255))
            
            image_train = cv2.cvtColor(image_train,cv2.COLOR_RGB2BGR)
            image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2BGR)
            st.sidebar.image(image_train)
            st.image(image1)

        else:
            st.subheader(result)

            image_train = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
            cv2.rectangle(image_train, (image_location_train[3],image_location_train[0]),
                               (image_location_train[1],image_location_train[2]), (0,255,0),2)

            cv2.putText(image_train,f"{result} {dst}",(60,60),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255))

            image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
    
            cv2.rectangle(image1, (image_location_test[3],image_location_test[0]),
                               (image_location_test[1],image_location_test[2]), (0,255,0),2)

            cv2.putText(image1,f"{result} {dst}",(60,60),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255))
            image_train = cv2.cvtColor(image_train,cv2.COLOR_RGB2BGR)
            image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2BGR)
            
            st.sidebar.image(image_train)
            st.image(image1)
    