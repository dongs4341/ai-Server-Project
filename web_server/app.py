from flask import Flask, request, render_template, jsonify
import requests

app = Flask(__name__)

AI_SERVER_URL = 'http://localhost:5001'  # AI 서버의 URL

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/send_image', methods=['POST'])
def send_image_to_ai_server():
    # 이미지 데이터를 받아옵니다.
    image_data = request.files['image'].read()

    # AI 서버로 이미지를 전송합니다.
    try:
        response_from_ai_server = requests.post(f'{AI_SERVER_URL}/process_image', files={'image': image_data})
        response_from_ai_server.raise_for_status()  # 오류가 있는 경우 예외 발생
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to connect to AI server: {str(e)}'}), 500

    if response_from_ai_server.status_code == 200:
        # AI 서버로부터 받은 숫자를 가져옵니다.
        extracted_number = response_from_ai_server.json().get('extracted_number', None)

        if extracted_number is not None:
            return jsonify({'extracted_number': extracted_number}), 200
        else:
            return jsonify({'error': 'Failed to extract number from AI server response.'}), 500
    else:
        return jsonify({'error': 'Failed to get response from AI server.'}), 500

if __name__ == '__main__':
    app.run(debug=True)