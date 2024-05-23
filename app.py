from flask import Flask, request, jsonify
import os
from pic2array import recognize

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # 保存文件到服务器
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # 使用 recognize 函数处理图片并获取结果
        result = recognize(filepath)

        # 删除文件（可选）
        os.remove(filepath)

        # 返回结果
        return jsonify({'result': result})


if __name__ == '__main__':
    # 确保 uploads 目录存在
    if not os.path.exists('uploads'):
        os.mkdir('uploads')
    app.run(debug=True)
