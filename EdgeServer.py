from ast import Str
from base64 import encode
from math import fabs
from multiprocessing.connection import wait
from re import T
import turtle
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# import connection library
import socketserver
import struct
import time
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


class MyTCPHandler (socketserver.BaseRequestHandler):

    def handle(self):

        framework = "tf"
        weights = "./checkpoints/yolov4/yolov4-416"
        size = 416
        tiny = False
        model = "yolov4"

        iou = 0.45
        score = 0.25
        dont_show = False

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        
        saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']


        while True:
            print("get...")
            data_stream = b''
            isRecceiving = True
            try:
                while isRecceiving:
                    data = self.request.recv(1024)
                    if not data:
                        print("Finish receiving")
                        break
                    data_stream += data

                    if data_stream[-2:] == b'\xff\xd9':
                        print("Finish receiving")
                        isRecceiving = False

            except Exception:
                print("\n\nKeep_receiving recv erro!\n\n")

            image = np.asarray(bytearray(data_stream), dtype="uint8")
            # print (image)
            image = cv2.imdecode (image, cv2.IMREAD_COLOR)
            # image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    

            # # loop through images in list and run Yolov4 model on each
            
            image_data = cv2.resize(image, (size, size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)


            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=iou,
                score_threshold=score
            )

            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            
            #~~~ Draw the bounding box
            imageResult = utils.draw_bbox(image, pred_bbox) 
            imageResult = Image.fromarray(imageResult.astype(np.uint8))
            # imageResult.show()

            #~~ Display result image by openCV
            imageResult = cv2.cvtColor(np.array(imageResult), cv2.COLOR_BGR2RGB)
            imageResult = cv2.resize(imageResult, (720, 960))
            cv2.namedWindow ("Image")    
            cv2.moveWindow("Image", 1000, 0)    
            cv2.imshow ("Image", imageResult)
            cv2.waitKey(100)

            # print (pred_bbox)


            #~~~ Convert result into string format
            # predBbox = ""
            # for i in range(pred_bbox[3][0]):
            #     predBbox += str(pred_bbox[0][0][i][0]) + " "
            #     predBbox += str(pred_bbox[0][0][i][1]) + " "
            #     predBbox += str(pred_bbox[0][0][i][2]) + " "
            #     predBbox += str(pred_bbox[0][0][i][3]) + " "
            #     predBbox += str(pred_bbox[1][0][i]) + " "
            #     predBbox += str(pred_bbox[2][0][i]) + " "

            # print (predBbox)

            #~~~ Send back detection result
            #
            # try:
            #     self.request.send(predBbox.encode())
            #     print("send back successful!")
            # except ConnectionResetError:
            #     print("ConnectionResetError client already disconnected, send 'results' back to client fail...")
            #     # Reconnect = True
            #     # continue
            # except BrokenPipeError:
            #     print("BrokenPipeError client already disconnected, send 'results' back to client fail...")
            #     # Reconnect = True           
            #     # continue


    def finish(self) -> None:
        print ("Finish")
        return super().finish()


if __name__ == '__main__':
    host = "192.168.0.106"
    port = 9851

    server = socketserver.TCPServer ((host, port), MyTCPHandler)
    server.serve_forever ()
