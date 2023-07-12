import cv2
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox


#capture = cv2.VideoCapture(0);
from vidgear.gears import CamGear

capture = CamGear(source='https://www.youtube.com/watch?v=01n6lkUS6vw', stream_mode=True,
                 logging=True).start()  # YouTube Video URL as input
#bkzhWLsgWb4
#"https://www.youtube.com/watch?v=bkzhWLsgWb4"
count = 0
#while capture.isOpened()==True:
while True:
    #_return, frame = capture.read()
    frame = capture.read()
    flip_gray = cv2.resize(frame, (1020, 600))

    count += 1
    if count % 6 != 0:
        continue


    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #flip image
    #flip_gray = cv2.flip(gray, 1)

    bbox, label, conf = cv.detect_common_objects(flip_gray)
    flip_gray = draw_bbox(flip_gray, bbox, label, conf)
    person_count = label.count("person")

    cv2.imshow("frame", flip_gray)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

"""
#person detectj
import cv2
import numpy as npu
import cvlib as cv
from cvlib.object_detection import draw_bbox


capture = cv2.VideoCapture(0);
count = 0

while capture.isOpened()==True:
    _return, frame = capture.read()
    count += 1
    if count % 6 != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #flip image
    flip_gray = cv2.flip(gray, 1)

    bbox, label, conf = cv.detect_common_objects(flip_gray)
    flip_gray = draw_bbox(flip_gray, bbox, label, conf)
    person_count = label.count("person")

    cv2.putText(flip_gray, f"Person- {person_count}", (50, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.imshow("frame", flip_gray)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()




"""



