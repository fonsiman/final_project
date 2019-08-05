import cv2



cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    model.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True))

    output = model.forward()[0,0,:,:]

    for detection in output:
        confidence = detection[2]
        if confidence > THRESHOLD:
            class_id = detection[1]
            class_name = class_names[class_id]
            # print(confidence, class_name)

            # rectangles!

            box_x=detection[3]
            box_y=detection[4]
            box_width=detection[5]
            box_height=detection[6]

            height, width, ch = frame.shape

            box_x = detection[3] * width
            box_y = detection[4] * height
            box_width = detection[5] * width
            box_height = detection[6] * height

            cv2.rectangle(frame,
                          (int(box_x),
                           int(box_y)),
                          (int(box_width),
                           int(box_height)),
                          (0, 0, 255),
                          thickness=2)

            cv2.putText(frame,
                        class_name + ' ' + str(round(confidence, 2)),
                        (int(box_x),
                         int(box_y+.05*height)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255))

    cv2.imshow('object_detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
