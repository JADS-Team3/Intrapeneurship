from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper function to get the next house name
def get_next_house_name():
    existing_houses = os.listdir(UPLOAD_FOLDER)
    house_numbers = [int(name.split()[-1]) for name in existing_houses if name.startswith("House")]
    next_number = max(house_numbers, default=0) + 1
    return f"House {next_number}"

# Main page
@app.route('/')
def index():
    return render_template('index.html')

# Upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            return "No file part"
        files = request.files.getlist('files[]')
        house_name = get_next_house_name()
        house_folder = os.path.join(app.config['UPLOAD_FOLDER'], house_name)
        os.makedirs(house_folder, exist_ok=True)
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(house_folder, filename))
        return redirect(url_for('gallery'))
    return render_template('upload.html')

# Gallery page
@app.route('/gallery')
def gallery():
    houses = os.listdir(UPLOAD_FOLDER)
    return render_template('gallery.html', houses=houses)

# House page
@app.route('/gallery/<house_name>')
def house(house_name):
    house_folder = os.path.join(app.config['UPLOAD_FOLDER'], house_name)
    if not os.path.exists(house_folder):
        return "House not found", 404

    # Gather all valid image files
    images = [f"uploads/{house_name}/{img}" for img in os.listdir(house_folder) if allowed_file(img)]
    
    # Calculate total number of images
    num_images = len(images)
    
    # Calculate total size of images (in MB)
    total_size_bytes = sum(
        os.path.getsize(os.path.join(house_folder, img)) for img in os.listdir(house_folder) if allowed_file(img)
    )
    total_size_mb = total_size_bytes / (1024 * 1024)  # Convert bytes to MB

    return render_template(
        'house.html',
        house_name=house_name,
        images=images,
        num_images=num_images,
        total_size_mb=total_size_mb
    )

if __name__ == '__main__':
    app.run(debug=True)
