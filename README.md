# Mobile_Object_Detection_with_Off-loading
嵌入式微處理器系統設計


## Android
- Local object detection by `MobileNetv1` through the Tensorflow Lite
- Input specification: 1*300*300*3 tensor input format transform from RGB bitmap 
- Output specification: 1*4*10 tensor output format with [RectF boundingBox, Category category, ...]
- Draw bounding box by a surface view and surface handler


## Remote Server
- Local object detection by `YoloV4` through the Tensorflow environment
- Based on https://github.com/hunglc007/tensorflow-yolov4-tflite
- Connection through socket TCP protocol
