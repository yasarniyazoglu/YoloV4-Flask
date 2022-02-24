from flask import Flask, render_template, Response
import argparse
from yolov4_flask import *


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()
    def get_frame(self):
        success, image = self.video.read()
        return image

app = Flask(__name__)

@app.route('/')  
def index():

    return render_template('index.html')

def yolov4_flask(camera):
    while True:
        frame = camera.get_frame()
        v4_inference(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')  
def video_feed():
    if model == 'yolov4_flask':
        return Response(yolov4_flask(VideoCamera()),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection using YOLO-Fastest in OPENCV')
    parser.add_argument('--model', type=str, default='yolov4_flask', choices=['yolov4_flask'])
    parser.add_argument('--semi-label', type=int, default=0, help="semi-label the frame or not")
    args = parser.parse_args()
    model = args.model
    app.run(host='0.0.0.0', debug=True, port=5000)
