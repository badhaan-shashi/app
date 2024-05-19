from flask import Flask, render_template, request, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from Rice_grain_Analysis import Img_thresholding

# Import the Google Drive upload function
from google_drive_upload import upload_file_to_drive

app = Flask(__name__)

# Configuration for the file upload
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No selected file'), 400

    if file:
        filename = secure_filename(file.filename)
        # Use a temporary directory for file storage
        temp_dir = tempfile.mkdtemp()
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)
        # git commit
        # Process the image with the analysis function
        results = Img_thresholding(filepath)
        print(results)
        
        # Upload the file to Google Drive
        image_url = upload_file_to_drive(filepath, filename)
        
        # Clean up the file from the temp directory after upload to avoid clutter
        os.remove(filepath)
        # os.rmdir(temp_dir)

        # Generate URL for the uploaded image on Google Drive or any static URL you can provide
        # image_url = f"URL_to_access_uploaded_image_on_Drive/{filename}"
        return jsonify(file_url=image_url, results=results, points=results[3])

    return jsonify(error='File upload failed'), 400

if __name__ == '__main__':
    app.run(debug=True)
