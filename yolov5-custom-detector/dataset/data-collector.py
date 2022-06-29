import cv2
import time
import pyautogui
import numpy as np
import pygetwindow as gw


def init_window(window_name):
    for wnd in gw.getWindowsWithTitle(window_name):
        if wnd.title == window_name:
            wnd.activate()
            return wnd


def read_window_frame(wnd):
    return np.array(
        pyautogui.screenshot(
            region=(wnd.left, wnd.top, wnd.width, wnd.height)
        )
    )[:, :, ::-1]


window = init_window("Sky2Fly")

while True:
    frame = read_window_frame(window)[75:-200, 30:-30, :]
    cv2.imshow("Screen", frame)

    code = cv2.waitKey(1)

    if code == ord("s"):
        filename = f"{int(time.time())}.png"
        cv2.imwrite(f"images/{filename}", frame)
        print(f"{filename} saved")
    elif code == 27:
        break
