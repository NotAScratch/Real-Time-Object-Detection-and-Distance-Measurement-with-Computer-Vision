from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
import tensorflow as tf

app = FastAPI()

# Load the pre-trained model
model = tf.saved_model.load("ssd_mobilenet_v3_large_coco_2020_01_14/saved_model")

# Load class names
with open('coco.names', 'r') as f:
    class_names = f.read().splitlines()

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    results = []
    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5:
            box = detection_boxes[i]
            class_name = class_names[detection_classes[i]]
            score = detection_scores[i]
            results.append({
                "class_name": class_name,
                "score": float(score),
                "box": box.tolist()
            })

    return {"detections": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
