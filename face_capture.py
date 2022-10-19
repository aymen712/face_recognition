import os
import mediapipe as mp
import cv2
#when the program run enter space 10 times
def face_capture():
    labe_number=len(os.listdir("training_data"))
    came = cv2.VideoCapture(1)
    image_counter=0
    os.mkdir(f"training_data\\s{labe_number}")
    os.chdir(f"training_data\\s{labe_number}")
    while True:
        ret,frame =came.read()
        cv2.imshow("cap_face",frame)
        k=cv2.waitKey(1)
        if image_counter !=10 and k%256==32 :
            cap_image = ("{}.png".format(image_counter))
            cap=cv2.imwrite(cap_image,frame)
            image_counter +=1
        if image_counter ==10 :
            labe_number += 1
            break
face_capture()

