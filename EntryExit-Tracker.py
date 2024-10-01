import cv2
from ultralytics import YOLO
import numpy as np
import torch
from tracker import Tracker


cap = cv2.VideoCapture(
    r"C:\Users\DELL\Downloads\Telegram Desktop\pythone\yoloV8\7390231-sd_426_240_25fps.mp4")

area = [(967, 200), (3, 200), (3, 230), (1100, 230)]

model = YOLO("yolov8n.pt")

tracker = Tracker()

passed_cars = set()

c = set()
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (0, 0), fx=3, fy=3)
    results = model(frame)
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 255), 3)
    points = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confidence = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = confidence[i]
            label = model.names[int(class_ids[i])]

            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)

            if "car" in label:
                points.append([x1, y1, x2, y2])
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                # cv2.putText(frame, str(label), (x1-20, y1),
                #             cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))

    boxes_id = tracker.update(points)
    # print(boxes_id)
    for box_id in boxes_id:
        x, y, w, h, idd = box_id
        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
        cv2.putText(frame, str(idd), (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        results = cv2.pointPolygonTest(np.array(area, np.int32), (w, h), False)
        # print(results)
        if results >= 0:
            c.add(idd)
    # print(c)
    a = len(c)
    cv2.putText(frame, 'number of people is ='+str(a), (50, 65),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

    cv2.imshow('FRAME', frame)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
