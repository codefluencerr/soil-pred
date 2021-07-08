# Importing essential libraries and modules

from flask import Flask, request, jsonify
# import pickle
# # from werkzeug.datastructures import FileStorage
# from PIL import Image as im
# from tensorflow import keras
# from flask import Flask, request, jsonify
# from keras.preprocessing import image
# import cv2
# import numpy as np
# # import time
# # import cloudinary
# import cloudinary.uploader
# import cloudinary.api
# from cloudinary.utils import cloudinary_url


# crop_recommendation_model_path = 'Agro-NBClassifier.pkl'
# crop_recommendation_model = pickle.load(
#     open(crop_recommendation_model_path, 'rb'))


app = Flask(__name__)


# @app.route('/crop_predict', methods=['GET'])
# def testt():
#     nitrogen = int(request.args.get("nitrogen"))
#     phosphorous = int(request.args.get('phosphorous'))
#     pottasium = int(request.args.get('pottasium'))
#     temperature = float(request.args.get('temperature'))
#     humidity = float(request.args.get('humidity'))
#     ph_level = float(request.args.get('ph_level'))
#     rainfall = float(request.args.get('rainfall'))
#     result = crop_recommendation_model.predict(
#         [[nitrogen, phosphorous, pottasium, temperature, humidity, ph_level, rainfall]])
#     print(result)
#     return jsonify({"result": str(result[0])}), 200


# ===============================================================================================
# # weed and soil project
# img_width, img_height = 150, 150
# model = keras.models.load_model('soil/SoilDetection_own1.h5')
app = Flask(__name__)


# labelsPath = 'weed/obj.names'
# print(labelsPath)
# # load weights and cfg
# weightsPath = 'weed/crop_weed_detection.weights'
# configPath = 'weed/crop_weed.cfg'
# LABELS = open(labelsPath).read().strip().split("\n")


# def predict_soil(path):
#     img_pred = image.load_img(path, target_size=(img_height, img_width))
#     img_pred = image.img_to_array(img_pred)
#     img = np.expand_dims(img_pred, axis=0)
#     res = model.predict_classes(img)
#     prob = model.predict_proba(img)
#     print('predicted class:', res)
#     print('predicted probability:', prob[0])
#     if res[0] == 0:
#         prediction = "Alluvial soil"
#     elif res[0] == 1:
#         prediction = "Black soil"
#     elif res[0] == 2:
#         prediction = "Clay soil"
#     else:
#         prediction = "Red soil"
#     print("Predicted Class", prediction)
#     return prediction


# def upload_file(file_to_upload):
#     file_loc = open('output.png', 'rb')
#     file = FileStorage(file_loc)
#     # file.save(dst='output1.png')
#     # file_loc.close()
#     print(f'------------------------------------------------->{type(file)}')
#     app.logger.info('in upload route')
#     cloudinary.config(cloud_name='dhyyf1dnu', api_key='366412935233217',
#                       api_secret='_mKRq5rxj23SmWkfROu9vX1yBpM')
#     upload_result = None
#     app.logger.info('%s file_to_upload', file)
#     upload_result = cloudinary.uploader.upload(file, folder='ml')
#     app.logger.info(upload_result)
#     app.logger.info(type(upload_result))
#     print(f"----------------------------->{ upload_result['url']}")
#     url = upload_result['url']
#     print(type(url))
#     return url


# def predict_weed(path):
#     # color selection for drawing bbox
#     np.random.seed(42)
#     COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
#     print("[INFO] loading YOLO from disk...")
#     net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#     # load our input image and grab its spatial dimensions
#     image = cv2.imread(path)
#     (H, W) = image.shape[:2]
#     # parameters
#     confi = 0.5
#     thresh = 0.5
#     # determine only the *output* layer names that we need from YOLO
#     ln = net.getLayerNames()
#     ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#     # construct a blob from the input image and then perform a forward
#     # pass of the YOLO object detector, giving us our bounding boxes and
#     # associated probabilities
#     blob = cv2.dnn.blobFromImage(
#         image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
#     net.setInput(blob)
#     start = time.time()
#     layerOutputs = net.forward(ln)
#     end = time.time()

#     # show timing information on YOLO
#     print("[INFO] YOLO took {:.6f} seconds".format(end - start))

#     # initialize our lists of detected bounding boxes, confidences, and
#     # class IDs, respectively
#     boxes = []
#     confidences = []
#     classIDs = []
#     # loop over each of the layer outputs
#     for output in layerOutputs:
#         # loop over each of the detections
#         for detection in output:
#             # extract the class ID and confidence (i.e., probability) of
#             # the current object detection
#             scores = detection[5:]
#             classID = np.argmax(scores)
#             confidence = scores[classID]

#             # filter out weak predictions by ensuring the detected
#             # probability is greater than the minimum probability
#             if confidence > confi:
#                 # scale the bounding box coordinates back relative to the
#                 # size of the image, keeping in mind that YOLO actually
#                 # returns the center (x, y)-coordinates of the bounding
#                 # box followed by the boxes' width and height
#                 box = detection[0:4] * np.array([W, H, W, H])
#                 (centerX, centerY, width, height) = box.astype("int")

#                 # use the center (x, y)-coordinates to derive the top and
#                 # and left corner of the bounding box
#                 x = int(centerX - (width / 2))
#                 y = int(centerY - (height / 2))

#                 # update our list of bounding box coordinates, confidences,
#                 # and class IDs
#                 boxes.append([x, y, int(width), int(height)])
#                 confidences.append(float(confidence))
#                 classIDs.append(classID)

#     # apply non-maxima suppression to suppress weak, overlapping bounding
#     # boxes
#     idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)

#     # ensure at least one detection exists
#     if len(idxs) > 0:
#         # loop over the indexes we are keeping
#         for i in idxs.flatten():
#             # extract the bounding box coordinates
#             (x, y) = (boxes[i][0], boxes[i][1])
#             (w, h) = (boxes[i][2], boxes[i][3])

#             # draw a bounding box rectangle and label on the image
#             color = [int(c) for c in COLORS[classIDs[i]]]
#             cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
#             text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
#             cv2.putText(image, text, (x+15, y + 35),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#     det = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     data = im.fromarray(det)
#     data.save('output.png')
#     result = upload_file('output.png')
#     return result
#     # cv2.imshow("image",image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()


@app.route('/soil', methods=['POST'])
def handle_soil():
    file_to_upload = request.files['file']
    print(type(file_to_upload))
    file_to_upload.save('input_file.jpg')
    # result = predict_soil('input_file.jpg')
    # return jsonify({"result": result}), 200
    return jsonify({"result": 'wola'}), 200


@app.route('/v1', methods=['POST'])
def handle_v1():
    return jsonify({"result": 'ss'}), 200


# @app.route('/weed', methods=['POST'])
# def handle_weed():
#     file_to_upload = request.files['file']
#     file_to_upload.save('input_file_weed.jpg')
#     result = predict_weed('input_file_weed.jpg')
#     return jsonify({"result": str(result)}), 200


if __name__ == '__main__':
    app.run(debug=True)
