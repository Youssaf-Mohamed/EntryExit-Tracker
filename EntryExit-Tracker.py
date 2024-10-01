import cv2
from ultralytics import YOLO
import numpy as np
from tracker import Tracker


def mouseEvents(event, x, y, flags, set):
    if event == cv2.EVENT_LBUTTONDOWN:
        XY = [x, y]
        print(XY)


cv2.namedWindow("frame")

# area 1

# [428, 584]
# [565, 651]
# [510, 670]
# [392, 594]


# area 2

# [361, 596]
# [474, 677]
# [427, 693]
# [309, 602]

######################################

people_entring = {}
entring = set()
people_exiting = {}
exiting = set()

######################################

tracker = Tracker()


cap = cv2.VideoCapture(
    r"C:\Users\DELL\Downloads\Telegram Desktop\pythone\openCv\vidoies\uhd_30fps.mp4")

area1 = [(428, 584), (565, 651), (515, 665), (392, 594)]
area2 = [(387, 598), (507, 668), (461, 681), (353, 607)]

model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 3)
    cv2.putText(frame, "1", (561, 684),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 3)
    cv2.putText(frame, "2", (498, 706),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if not ret:
        break

    results = model(frame)

    points = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confidence = r.boxes.conf.cpu().numpy()
        class_name = r.boxes.cls.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            cls = model.names[int(class_name[i])]

            conf = confidence[i]

            cx = int(x1 + x2)
            cy = int(y1 + y2)

            if "person" in cls:

                points.append([x1, y1, x2, y2])

    boxes_id = tracker.update(points)

    for box_id in boxes_id:
        x, y, w, h, id = box_id

        result1 = cv2.pointPolygonTest(
            np.array(area2, np.int32), ((w, h)), False)

        if result1 >= 0:
            people_entring[id] = (w, h)
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)

        if id in people_entring:
            result2 = cv2.pointPolygonTest(
                np.array(area1, np.int32), ((w, h)), False)
            if result2 >= 0:
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                cv2.circle(frame, (x, h), 5, (0, 0, 0), -1)
                cv2.putText(frame, "Person", (x, y - 10),
                            cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 255), 2)
                cv2.putText(frame, f"  id= {
                            id}", (x+65, y - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 2)
                entring.add(id)

        result3 = cv2.pointPolygonTest(
            np.array(area1, np.int32), ((w, h)), False)

        if result3 >= 0:
            people_exiting[id] = (w, h)
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)

        if id in people_exiting:
            result4 = cv2.pointPolygonTest(
                np.array(area2, np.int32), ((w+x//2, h)), False)
            if result4 >= 0:
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                cv2.circle(frame, (x, h), 4, (0, 0, 0), -1)
                cv2.putText(frame, "Person", (x, y - 10),
                            cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 255), 1)
                cv2.putText(frame, f"  id={
                            id}", (x+55, y - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                exiting.add(id)

    i = len(entring)
    o = len(exiting)

    cv2.putText(frame, 'Number of entering people= '+str(i),
                (20, 44), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'Number of exiting people= '+str(o),
                (20, 82), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("frame", frame)
    cv2.setMouseCallback("frame", mouseEvents)

    if cv2.waitKey(1) and 0xff == ord("q"):
        break


cv2.destroyAllWindows()
cap.release()
