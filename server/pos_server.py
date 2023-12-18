import argparse
from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)



@app.route('/')
def index():
    image_folder = os.path.join(app.root_path, impath)
    image_list = sorted(os.listdir(image_folder))

    model_file=impath.split("/")[-2]
    print("asdfasdf ", model_file)
    
    return render_template('index.html', image_list=image_list, num_images=len(image_list), model_file=model_file)

@app.route('/images/<filename>')
def get_image(filename):
    image_folder = os.path.join(app.root_path, impath)
    return send_from_directory(image_folder, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Your program description')
    parser.add_argument('--impath', type=str, help='Path to the image directory')
    args = parser.parse_args()

    if args.impath:
        impath = args.impath
    else:
        impath = "/home/wubonacci/racecar_gym_competition_rl/record/cppo1_img"
    
    
    
    app.run(port=0, debug=True)
