import cv2
import pyautogui
import numpy as np
from Detector import Detector


if __name__ == '__main__':
    out = cv2.VideoWriter(
        "readme-assets/result-video.avi",
        cv2.VideoWriter_fourcc(*"MJPG"), 5, (800, 600)
    )
    detector = Detector(model_name="best.pt")

    while True:
        frame = np.array(pyautogui.screenshot())[75:-200, 30:1000, :]

        # print(detector.detect(frame))
        print(detector.detect(frame, show=True))

        if cv2.waitKey(1) & 0xFF == 27:
            break

    out.release()
