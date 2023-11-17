# Mobile_Object_Detection_with_Off-loading
嵌入式微處理器系統設計


## Android
- Local object detection by `MobileNetv1` through the Tensorflow Lite
- Input specification: $1 * 300 * 300 * 3$ tensor input format transform from RGB bitmap 
- Output specification: $1 * 4 * 10$ tensor output format with [RectF boundingBox, Category category, ...]
- Draw bounding box by a surface view and surface handler


## Remote Server
- Local object detection by `YoloV4` through the Tensorflow environment
- Based on https://github.com/hunglc007/tensorflow-yolov4-tflite
- Input specification: $1 * 416 * 416 * 3$ tensor input format transform from RGB bitmap 
- Output specification: $1 * 4 * 10$ tensor output format with [RectF boundingBox, float accuracy, int class, int validation]
- Connection through socket TCP protocol

## Demo
1. You can choose the object detection mode by the navigation bar at the bottom of the APP
   ![image](https://github.com/KunSheng-Liu/Mobile_Object_Detection_with_Off-loading/assets/65661575/5dc3f893-f415-40b0-860c-56247d078cab)
2. As you can see, local detection has poor accuracy compared to remote detection under this complex view
   ![image](https://github.com/KunSheng-Liu/Mobile_Object_Detection_with_Off-loading/assets/65661575/47d8cf21-dfd8-4209-a7fd-703e707168af)
