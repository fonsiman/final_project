import cv2

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()
# fgbg = cv2.createBackgroundSubtractorKNN()

roi_start = 50
roi_size = 200
kernel = np.ones((3, 3), np.uint8)

classes = {'01_palm': 0,
           '02_l': 1,
           '03_fist': 2,
           '04_fist_moved': 3,
           '05_thumb': 4,
           '06_index': 5,
           '07_ok': 6,
           '08_palm_moved': 7,
           '09_c': 8,
           '10_down': 9}

classes_reverse = {v:k for k,v in classes.items()}

while True:
    ret, frame = cap.read()

    w = frame.shape[1]
    h = frame.shape[0]
    hcenter = w // 2
    vcenter = h // 2


    roi = frame[roi_start:roi_start+roi_size,
                roi_start:roi_start+roi_size]

    #hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # define range of skin color in HSV
    #lower_skin = np.array([0,20,70], dtype=np.uint8)
    #upper_skin = np.array([20,255,255], dtype=np.uint8)

    #extract skin colur imagw
    #mask = cv2.inRange(hsv, lower_skin, upper_skin)

    #extrapolate the hand to fill dark spots within
    #mask = cv2.dilate(mask, kernel, iterations = 4)

    #blur the image
    #mask = cv2.GaussianBlur(mask, (5, 5), 100)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray_cropped = gray[:, (hcenter - h // 2):(hcenter + h // 2)]

    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    model_input = np.expand_dims(np.expand_dims(cv2.resize(mask, (IMG_SIZE, IMG_SIZE)), axis=2), axis=0) / 255

    prediction = model.predict_proba(model_input)

    class_name = classes_reverse[np.argmax(prediction)]
    confidence = np.max(prediction)

    cv2.rectangle(frame, (roi_start, roi_start), (roi_start+roi_size, roi_start+roi_size), (0, 255, 0) ,0)

    cv2.putText(frame,
                class_name + ' ' + str(confidence) if confidence > 0.5 else 'good luck!',
                (int(roi_start),
                 int(roi_start + .09 * roi_size)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255))

    cv2.imshow('my_webcam_frame', frame)
    cv2.imshow('my_webcam_proc', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
