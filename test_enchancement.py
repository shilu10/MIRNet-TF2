lowlight_test_images_path = 'lol_dataset/eval15/low/'


for test_file in glob.glob(lowlight_test_images_path + "*.png"):
    print(test_file.split("/")[-1])
    filename = test_file.split("/")[-1]
    data_lowlight_path = test_file
    original_img = Image.open(data_lowlight_path)
    original_size = (np.array(original_img).shape[1], np.array(original_img).shape[0])
    original_img = (np.asarray(original_img)/255.0)

    img_lowlight = Image.open(data_lowlight_path)
    img_lowlight = (np.asarray(img_lowlight)/255.0) 
    img_lowlight = np.expand_dims(img_lowlight, 0)
    img_lowlight = tf.convert_to_tensor(img_lowlight, dtype=tf.float32)

    enhanced_image = model(img_lowlight)
 
    enhanced_image = tf.cast((enhanced_image[0,:,:,:] * 255), dtype=np.uint8)
    enhanced_image = Image.fromarray(enhanced_image.numpy())
    enhanced_image.save(f"results/enchancement/{filename}")
