import torch
import cv2
import os
import numpy as np
from ultralytics import YOLOv10
from time import time


class YoloModel:
    """
    Class to load and use a YOLO v10 model to detect objects in images
    """
    def __init__(self, name: str, version: str, online: bool = False, device: str = 'cpu'):
        """
        Load the model from the .pt file located in the model folder
        :name: name of the model to load
        :version: version of the model to load 'best' or 'last'
        :online: if it should download the model from the internet
        """

        # current dir
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        # if online download the model
        if online:
            self.model = self.load_model()
        else:
            # check if name and version are valid
            if name not in os.listdir(self.current_dir + '/model'):
                raise ValueError(f"Model {name} not found")
            if version + '.pt' not in os.listdir(self.current_dir + f'/model/{name}/weights'):
                raise ValueError(f"Version {version} not found")
            
            model_path = self.current_dir + f'/model/{name}/weights/{version}.pt'
            self.model = self.load_model(model_path)

        # get the classes       
        self.classes = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.device = device
        self.model.to(self.device)
        print(f"Using {self.device} device")

    def load_model(self, model_path=None):
        """
        Load the model from the disk
        :param model_path: name of the model to load
        """
        if model_path is not None:
            home_path = os.path.expanduser("~")
            yolo_path = home_path + '/yolov5'
            model = torch.hub.load(yolo_path, 'custom', source='local', path=model_path, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        return model
        
    def score_frame(self, frame):
        """
        Score the frame with the model
        :param frame: frame to score
        :return: frame with the objects detected
        """
        # score the frame
        results = self.model(frame)
        # Convert results to dictionarie per detection
        results_list = results.pandas().xyxy[0].to_dict('records')
        image = results.render()
        # print results pretty
        for i in results_list:
            print(i)

        return results_list, image


if __name__ == '__main__':
    # current dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = current_dir + '/model/epoch_100/weights/best.pt'
    model = YoloModel(name='epoch_100', version='best')

    # Load image wih pil
    img = cv2.imread(current_dir + '/dataset/IMG-20240821-WA0023.jpg')
    # Score the image
    results, img = model.score_frame(img)
