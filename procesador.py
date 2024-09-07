import os
import cv2
import supervision as sv
from ultralytics import YOLOv10

class Procesador:
    """
    Class to load and use a YOLO v10 model to detect objects in images
    """
    def __init__(self):
        """
        Load the model from the .pt file located in the model folder
        :name: name of the model to load
        :version: version of the model to load 'best' or 'last'
        :online: if it should download the model from the internet
        """
        # define HOME_PATH
        self.HOME_PATH = os.path.dirname(os.path.abspath(__file__))
        self.models = os.listdir(os.path.join(self.HOME_PATH, 'model'))
        self.weights_path = {}
        self.classes = {}
        self.class_colors = {}
        self.model = None
        # Load variables
        self.init_load()

    def init_load(self):
        """
        Load and define basic parameters
        """
        for model in self.models:
            self.weights_path[model] = {}
            for weight in os.listdir(os.path.join(self.HOME_PATH, 'model', model, 'weights')):
                w_key = weight.split('.')[0]
                self.weights_path[model][w_key] = os.path.join(self.HOME_PATH, 'model', model, 'weights', weight)

        # define colors per classes : Temporal it has to be automated
        self.class_colors = {
            'desordenado': (255, 0, 0),  # Rojo
            'ordenado': (0, 255, 0),  # Verde
        }

    def load_model(self, model_path=None):
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            self.model = YOLOv10(model_path)
            return True
        else:
            print("File not found: ", model_path)
            return False
    
    def draw_rect(self, image, x1, y1, x2, y2, color=(255, 0, 0), thickness=5):
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return image

    # Dibuja un texto en la imagen
    def draw_text(self, image, text, x, y, color=(255, 0, 0), thickness=2):
        # poner en mayúsculas
        text = text.upper()
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, thickness)
        top_left = (int(x), int(y) - text_height - baseline)
        bottom_right = (int(x) + text_width, int(y) + baseline)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        
        cv2.putText(image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness)
        return image

    
    def anotate_frame(self, detections, results, image_res):
        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[i]
            image_res = self.draw_rect(image_res, x1, y1, x2, y2, color=self.class_colors[results.names[detections.class_id[i]]])
            image_res = self.draw_text(image_res, results.names[detections.class_id[i]], 10, 50, color=self.class_colors[results.names[detections.class_id[i]]])

        return image_res

    def score_frame(self, frame):
        """
        Score the frame with the model
        :param frame: frame to score
        :return: frame with the objects detected
        """
        # Pre process the frame
        # detect orientation of image and rotate if necessary
        if frame.shape[0] > frame.shape[1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # make sure the image is in RGB if not convert it
        if frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        # resize image
        frame = cv2.resize(frame, (1280, 960))

        if self.model:
            image_results = self.model(frame)[0]
            detections = sv.Detections.from_ultralytics(image_results)
            # anotate the image
            image_anotated = self.anotate_frame(detections, image_results, frame)
            return image_anotated, detections
        else:
            print("Model not loaded")
            return frame
            


if __name__ == '__main__':
    # define the model
    procesador = Procesador()
    print(procesador.HOME_PATH)
    # terminar el programa
    print("Done!")

    # print pretty dictionary
    for model in procesador.weights_path:
        print(model)
        for weight in procesador.weights_path[model]:
            print(f'\t{weight}: {procesador.weights_path[model][weight]}')

    # Load the model
    model_path = procesador.weights_path['epoch_70']['best']
    procesador.load_model(model_path)
    # Load the image
    image = cv2.imread(r'D:\Personal\Noodle_detector\dataset\IMG-20240821-WA0021.jpg')
    print(image.shape, type(image))
    # Score the image
    img, results = procesador.score_frame(image)
    # save the image
    cv2.imwrite('results.jpg', img)
    print(img.shape, type(img))
    print(results)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



