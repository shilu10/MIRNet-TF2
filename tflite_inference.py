from PIL import Image
import numpy as np
import tensorflow as tf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--tflite_model_path', type=str, default='checkpoint/saved/super_resolution/best_model.tflite')
parser.add_argument('--test_data_path', type=str, default='test/LIME')
parser.add_argument('--result_data_path', type=str, default='results/tflite/LIME')


args = parser.parse_args()

def inferrer(image):
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    interpreter.set_tensor(input_index, input_image)

    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    # Convert output array to image
    output_image = (np.squeeze(output, axis=0).clip(0, 1) * 255).astype(np.uint8)

    img = Image.fromarray(output_image)


def test():

    lowlight_test_images_path = args.test_data_path

    for test_file in glob.glob(lowlight_test_images_path + "*.png"):
        filename = test_file.split("/")[-1]
        data_lowlight_path = test_file
        original_img = Image.open(data_lowlight_path)
        original_size = (np.array(original_img).shape[1], np.array(original_img).shape[0])
        original_img = cv2.resize(original_img, (256, 256))

        img_lowlight = cv2.imread(data_lowlight_path)
        img_lowlight = cv2.resize(img_lowlight, (256, 256))
        
        y = img_to_array(img_lowlight)
        inputs = np.expand_dims(y, axis=0)
        t = time.time()
        enhanced_image = inferrer(inputs)
        print(time.time() - t)


        if args.plot_results:
            plt.figure()
            plt.subplot(121)
            plt.imshow(original_img/255.0)
            
            plt.subplot(122)
            plt.imshow(enhanced_image/255.0)
        
        save_file_dir = lowlight_test_images_path.replace('test', 'results')
        save_file_path = save_file_dir + "/" + filename
        enhanced_image.save(fsave_file_path)


if __name__ == '__main__':
    test()