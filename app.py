# Importing essential libraries and modules

from flask import Flask, request, jsonify
from tensorflow import keras
from flask import Flask, request, jsonify
from keras.preprocessing import image
import numpy as np


# crop_recommendation_model_path = 'Agro-NBClassifier.pkl'
# crop_recommendation_model = pickle.load(
#     open(crop_recommendation_model_path, 'rb'))


app = Flask(__name__)


# ===============================================================================================
# # weed and soil project
img_width, img_height = 150, 150
model = keras.models.load_model('soil/SoilDetection_own1.h5')
app = Flask(__name__)


def predict_soil(path):
    img_pred = image.load_img(path, target_size=(img_height, img_width))
    img_pred = image.img_to_array(img_pred)
    img = np.expand_dims(img_pred, axis=0)
    res = model.predict_classes(img)
    prob = model.predict_proba(img)
    print('predicted class:', res)
    print('predicted probability:', prob[0])
    if res[0] == 0:
        prediction = "Alluvial soil"
    elif res[0] == 1:
        prediction = "Black soil"
    elif res[0] == 2:
        prediction = "Clay soil"
    else:
        prediction = "Red soil"
    print("Predicted Class", prediction)
    return prediction


@app.route('/soil', methods=['POST'])
def handle_soil():
    file_to_upload = request.files['file']
    print(type(file_to_upload))
    file_to_upload.save('input_file.jpg')
    result = predict_soil('input_file.jpg')
    return jsonify({"result": result}), 200
    # return jsonify({"result": 'wola'}), 200


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
