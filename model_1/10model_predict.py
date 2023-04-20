from tensorflow import keras
import os
import numpy as np
from PIL import Image

model = keras.models.load_model('my_model.h5')



# Create a list to store the predicted results
predicted_results = []

# specify the folder path
folder_path = "01_init\\train_test_data\\test\\cefazolin"

# get a list of all the files in the folder with a .jpg extension
file_list = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# print the path to each file in the list
for file in file_list:
    file_path = os.path.join(folder_path, file)
    # print(file_path)
    # Load a single image for prediction
    image_path = file_path
    image = Image.open(image_path)
    image = image.resize((150, 300))
    image_array = np.array(image)
    image_array = image_array.reshape((1, 150, 300, 1))

    # Make a prediction using the trained model
    prediction = model.predict(image_array)

    # Convert the prediction from a probability distribution to a class label
    predicted_class = np.argmax(prediction)

    # Print the predicted class label
    if predicted_class == 0:
        f = open("predicted.txt", "a")
        f.write("amikacin\n")
        f.close()
        print("amikacin")
    else:
        f = open("predicted.txt", "a")
        f.write("cefazolin\n")
        f.close()
        print("cefazolin")
# # Load a single image for prediction
# image_path = "01_init\\train_test_data\\test\\amikacin\\amikacin_29.png"
# image = Image.open(image_path)
# image = image.resize((150, 300))
# image_array = np.array(image)
# image_array = image_array.reshape((1, 150, 300, 1))

# # Make a prediction using the trained model
# prediction = model.predict(image_array)

# # Convert the prediction from a probability distribution to a class label
# predicted_class = np.argmax(prediction)

# # Print the predicted class label
# if predicted_class == 0:
#     f = open("predicted.txt", "a")
#     f.write("amikacin")
#     f.close()
# else:
#     f = open("predicted.txt", "a")
#     f.write("cefazolin")
#     f.close()




# # Iterate over the images in X_test
# for i in range(len(X_test)):
#     image = X_test[i]
#     image_array = image.reshape((1, 150, 300, 1))
#     prediction = model.predict(image_array)
#     predicted_class = np.argmax(prediction)
#     if predicted_class == 0:
#         predicted_result = "Amikacin"
#     else:
#         predicted_result = "Cefazolin"
#     predicted_results.append(predicted_result)
    
# # Write the predicted results to a CSV file
# with open('predictions.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Image Index', 'Predicted Result'])
#     for i in range(len(predicted_results)):
#         writer.writerow([i, predicted_results[i]])