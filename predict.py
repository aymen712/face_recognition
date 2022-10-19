import os
import mediapipe as mp
import cv2
import numpy as np
def face_recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    ###########aymen_recognition##################################################################################################################
    mpfacedetection = mp.solutions.face_detection
    mpdraw = mp.solutions.drawing_utils
    facedetection = mpfacedetection.FaceDetection(0.7)
    path = "training_data"
    i = 0
    faces = []
    list_libel=[]
    faceID = []
    dirs = os.listdir(path)
    for dir_name in dirs:
        label1 = int(dir_name.replace("s", ""))
        subject_dir_path = path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            image_path = subject_dir_path + "/" + image_name
            frame1 = cv2.imread(image_path)
            imgRGB = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
            imgRGB2 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            result = facedetection.process(imgRGB)
            image_array = np.array(imgRGB2)
            if result.detections:
                for id, detection in enumerate(result.detections):
                    mpdraw.draw_detection(frame1, detection)
                    bboxc = detection.location_data.relative_bounding_box
                    ih, iw, ic = frame1.shape
                    bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), \
                        int(bboxc.width * iw), int(bboxc.height * ih)
                    cv2.rectangle(frame1, bbox, (255, 0, 255), 2)
                    confidence = image_array[int(bboxc.ymin * ih):int(bboxc.ymin * ih) + int(bboxc.width * iw),
                        int(bboxc.xmin * iw):int(bboxc.xmin * iw) + int(bboxc.height * ih)]
                    faces.append(confidence)
                    faceID.append(label1)
    recognizer.train(faces, np.array(faceID))
    recognizer.save('Trainner.yml')
##############detect_face###########################################################################################
    came = cv2.VideoCapture(1)
    while True:
        ret, frame = came.read()
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        imgRGB2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        result = facedetection.process(imgRGB)
        image_array1 = np.array(imgRGB2)
        if result.detections:
            for id, detection in enumerate(result.detections):
                mpdraw.draw_detection(frame, detection)
                bboxc = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), \
                       int(bboxc.width * iw), int(bboxc.height * ih)
                image_array1[int(bboxc.ymin * ih):int(bboxc.ymin * ih) + int(bboxc.width * iw),\
                            int(bboxc.xmin * iw):int(bboxc.xmin * iw) + int(bboxc.height * ih)]
                cv2.rectangle(frame, bbox, (255, 0, 255), 6)
                face_detect_live = image_array1[int(bboxc.ymin * ih):int(bboxc.ymin * ih) + int(bboxc.width * iw),
                     int(bboxc.xmin * iw):int(bboxc.xmin * iw) + int(bboxc.height * ih)]
                id1, _ = recognizer.predict(face_detect_live)
                print(id1)
                print(_)
                if _ < 50:
                    print(id1)
                    print("true")
                else:
                    cv2.rectangle(frame, bbox, (255, 0, 444), 5)
                    print("false")
            cv2.imshow("fff",frame)
            cv2.waitKey(1)
face_recognition()

