"""This is the main entrypoint of this project.
    When executed as
    python main.py input_folder/
    it should perform the neural network inference on all images in the `input_folder/` and print the results.
"""

from argparse import ArgumentParser
import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def get_model(model_file="model.tflite"):
    """
        Function to get tflite interpreter (model)
    """
    return tf.lite.Interpreter(model_path=model_file)


def get_images_and_names(folder, shape):
    """
        Return numpy images with shape (n,h,w,c) and list of image's names from folder
        n: Number of images
        h: Images height
        w: Images Width
        c: Channels
    """
    folder_files = os.listdir(folder)
    image_files = [image_file for image_file in folder_files 
                   if image_file.split('.')[-1].lower() in ["jpg", "png", "jpeg", "jfif"]]
    list_images = [np.asarray(Image.open(os.path.join(folder, image)).resize((shape[1], shape[0])), dtype=np.float32) 
                   for image in image_files]

    return np.array(list_images), image_files


def get_results(outputs, names):
    """
        Get for all predicted images, score confidence and class predicted
    """
    results = []
    for i, output in enumerate(outputs):
        proba_output = tf.nn.softmax(output).numpy()
        best_index = proba_output.argsort()[-1]
        result = {names[i]:{"score": proba_output[best_index], "class": class_names[best_index]}}
        results.append(result)
    return results

def parse_args():
    """Define CLI arguments and return them.

    Feel free to modify this function if needed.

    Returns:
        Namespace: parser arguments
    """
    parser = ArgumentParser()
    parser.add_argument("input_folder", type=str, help="A folder with images to analyze.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    # do stuff ...

    interpreter = get_model()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data, image_names = get_images_and_names(cli_args.input_folder, tuple(input_details[0]["shape"][1:3]))

    images_number = input_data.shape[0]

    interpreter.resize_tensor_input(input_details[0]["index"], [images_number, *input_details[0]["shape"][1:]])
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    results = get_results(output_data, image_names)

    print(f"Analyzing folder : {cli_args.input_folder}", file=sys.stderr)  # info print must be done on stderr like this for messages

    # print results on stdout
    for result in results:
        print(result, file=sys.stdout)