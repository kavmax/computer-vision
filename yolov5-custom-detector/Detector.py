import cv2
import time

import pyautogui
import torch
# import torchaudio
# import torchvision
import numpy as np


class Detector:
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using Device: {self.device}")

    @staticmethod
    def load_model(model_name):
        """
        Loads Yolov5 model from Pytorch hub.
        :param model_name:
        :return: Trained Pytorch model.
        """
        return torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=model_name,
            force_reload=False
        )

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolov5 model
        :param frame: input frame in numpy/list/tuple format.
        :return: labels and coordinates of objects detected by model in the frame
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def interpret_results(self, results, frame):
        final_results = []
        labels, cord = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for idx, label in enumerate(labels):
            row = cord[idx]
            if row[4] >= 0.5:
                x1, y1 = int(row[0]*x_shape), int(row[1]*y_shape)
                x2, y2 = int(row[2]*x_shape), int(row[3]*y_shape)

                final_results.append([
                    self.class_to_label(label), x1, y1, x2, y2
                ])

        return final_results

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on the frame
        :param results: contains labels and coordinates predicted by model on the given frame
        :param frame: frame which has been scored
        :return: frame with bounding boxes and labels plotted on it
        """
        for result in results:
            class_name, x1, y1, x2, y2 = result
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 7)

        return frame

    def detect(self, image, show=False):
        results = self.interpret_results(self.score_frame(image), image)

        if show:
            image = self.plot_boxes(results, image)
            image_resized = cv2.resize(image, (800, 600))
            cv2.imshow("frame", image_resized)
            cv2.waitKey(1)

        return results
