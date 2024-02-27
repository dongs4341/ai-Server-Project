from flask import Flask, request, jsonify
import argparse

app = Flask(__name__)
#models = {} 

@app.route('/')
def index():
    return "AI Server is running!"


DETECTION_URL = "/v1/object-detection/<model>"  # <model>에 실제 모델이름이 들어감

@app.route(DETECTION_URL, methods=["POST"])
def predict(model):
    if request.method != "POST":
        return jsonify({'error': 'Invalid request method. Please use POST.'}), 405
    
    # 이미지 처리 및 모델 사용 코드(생략)
    # 대신 단순히 고정된 값을 반환
    return jsonify({'extracted_number': 999999}), 200

@app.route("/process_image", methods=["POST"])
def upload_barcode():
    # 성공 메시지 반환
    return jsonify({'message': '바코드 이미지가 성공적으로 받아졌습니다', 'extracted_number': 999999}), 200

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API for testing")
    parser.add_argument("--port", default=5001, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)     # 어떤 호스트에서도 연결 가능 //배포시 사용 X
    # 기본적으로 제공되는 내장 웹서버

'''사진 받는 거 합침
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import pytesseract
import argparse
import traceback  # 추가

app = Flask(__name__)
models = {}

DETECTION_URL = "/process_image"

def process_image_from_request(request):
    if "image" in request.files:
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        return im
    return None

@app.route(DETECTION_URL, methods=["POST"])
def process_image():
    try:
        im = process_image_from_request(request)

        if im:
            model_name = 'yolov5s'
            if model_name in models:
                results = models[model_name](im, size=640)
                bbox = results.xyxy[0].cpu().numpy()

                barcode_part = im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

                extracted_number = pytesseract.image_to_string(barcode_part, config='--psm 8 digits')

                return jsonify({'extracted_number': extracted_number}), 200
    except Exception as e:
        # 에러 메시지를 로깅
        print(f"Error processing image: {str(e)}")
        traceback.print_exc()

    return jsonify({'error': 'Failed to process image.'}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5001, type=int, help="port number")
    parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    for m in opt.model:
        models[m] = torch.hub.load("ultralytics/yolov5", m, 'best.pt', force_reload=True, skip_validation=True)

    app.run(host="0.0.0.0", port=opt.port)
'''


'''//이제 진짜
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import pytesseract
import argparse  # 추가

app = Flask(__name__)
models = {}

DETECTION_URL = "/v1/object-detection/<model>"

def process_image_from_request(request):
    if "image" in request.files:
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        return im
    return None

@app.route(DETECTION_URL, methods=["POST"])
def predict(model):
    if request.method != "POST":
        return jsonify({'error': 'Invalid request method. Please use POST.'}), 405

    im = process_image_from_request(request)
    if im is None:
        return jsonify({'error': 'No image file provided.'}), 400

    if model in models:
        # YOLOv5 객체 탐지 수행
        results = models[model](im, size=640)
        bbox = results.xyxy[0].cpu().numpy()  # 바운딩 박스 추출

        # 추출된 바코드 숫자 부분
        barcode_part = im.crop((bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]))

        # OCR을 사용하여 숫자 추출
        extracted_number = pytesseract.image_to_string(barcode_part, config='--psm 8 digits')

        # 추출된 숫자를 웹서버로 전송
        return jsonify({'extracted_number': extracted_number}), 200
    else:
        return jsonify({'error': 'Model not found.'}), 404

@app.route("/upload_barcode", methods=["POST"])
def upload_barcode():
    im = process_image_from_request(request)
    if im:
        im.save("received_barcode.jpg")  # 바코드 이미지 저장
        return jsonify({'message': '바코드 이미지가 성공적으로 받아졌습니다'}), 200
    else:
        return jsonify({'error': 'Failed to process image.'}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5001, type=int, help="port number")
    parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    for m in opt.model:
        # 모델을 캐시에서 로드하도록 설정 변경 고려
        models[m] = torch.hub.load("ultralytics/yolov5", m, pretrained=True)
        #models[m] = torch.hub.load("ultralytics/yolov5",  m, 'best.pt', force_reload=True, skip_validation=True) //모델 로딩 시 force_reload=True는 개발 중에 유용할 수 있지만, 운영 환경에서는 필요하지 않을 수 있습니다. 모델을 캐시에서 로드하도록 설정하여 로딩 시간을 단축할 수 있음(선택사항)
    

        ####실제는 이런식으로 함
        # 팀원이 제공한 가중치 파일 경로
        #model_path = 'path/to/yolov5/weights.pt'  # 실제 경로로 변경해야 합니다.
        # 모델 로드
        #model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    
    app.run(host="0.0.0.0", port=opt.port)
    '''