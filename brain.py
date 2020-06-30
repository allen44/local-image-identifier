
#Define the execution path
import os
execution_path = os.getcwd()
print(execution_path)

#Add your own photo to identify
from sys import argv
try:
    image_filename = argv[1]
except:
    image_filename = 'giraffe.jpg'

#Models folder
models_folder = os.environ["USERPROFILE"] + "\\Models\\"
models_filenames = os.listdir(models_folder)
models_abs_paths = []
for model_filename in models_filenames:
    models_abs_paths.append(os.path.join(models_folder, model_filename))
print(f"models_folder: {models_folder}")
print(f"models_abs_paths: {models_abs_paths}")

#Photos folder
image_folder = execution_path
allfiles = os.listdir(image_folder)
images_files = []
for eachPicture in allfiles:
    if eachPicture.endswith(".png") or eachPicture.endswith(".jpg"):
        images_files.append(eachPicture)
print(images_files)

#Initialize the Model
from imageai.Prediction import ImagePrediction
result_count = 1

for model_abs_path in models_abs_paths:
    if 'squeezenet' in model_abs_path.lower():
        try:
            prediction = ImagePrediction()
            prediction.setModelTypeAsSqueezeNet()
            prediction.setModelPath(model_abs_path)
            prediction.loadModel()
            for image_file in images_files:
                predictions, probabilities = prediction.predictImage(os.path.join(execution_path, image_file), result_count=result_count )
                print(f"\n\nThe SqueezeNet model's top {result_count} predictions with probability for {image_file}: ")
                for eachPrediction, eachProbability in zip(predictions, probabilities):
                    print(eachPrediction , " : " , eachProbability)
        except Exception as e:
            print("\n\nSqueezeNet didn't work.")
            print(e)
            print("Next model..")
    elif 'resnet' in model_abs_path.lower():
        try:    
            prediction2 = ImagePrediction()
            prediction2.setModelTypeAsResNet()
            prediction2.setModelPath(model_abs_path)
            prediction2.loadModel()
            for image_file in images_files:
                predictions2, probabilities2 = prediction2.predictImage(os.path.join(execution_path, image_file), result_count=result_count )
                print(f"\n\nThe ResNet model's top {result_count} predictions with probability for {image_file}: ")
                for eachPrediction2, eachProbability2 in zip(predictions2, probabilities2):
                    print(eachPrediction2 , " : " , eachProbability2)
        except Exception as e:
            print("\n\n\ResNet didn't work.")
            print(e)
            print("Next model..")
    elif 'inception' in model_abs_path.lower():
        try:
            prediction3 = ImagePrediction()
            prediction3.setModelTypeAsInceptionV3()
            prediction3.setModelPath(model_abs_path)
            prediction3.loadModel()
            for image_file in images_files:
                predictions3, probabilities3 = prediction3.predictImage(os.path.join(execution_path, image_file), result_count=result_count )
                print(f"\n\nThe Inception model's top {result_count} predictions with probability for {image_file}: ")
                for eachprediction3, eachProbability3 in zip(predictions3, probabilities3):
                    print(eachprediction3 , " : " , eachProbability3)
        except Exception as e:
            print("\n\ninception didn't work.")
            print(e)
            print("Next model..")
    elif 'densenet' in model_abs_path.lower():
        try:
            prediction4  = ImagePrediction()
            prediction4.setModelTypeAsDenseNet()
            prediction4.setModelPath(model_abs_path)
            prediction4.loadModel()
            for image_file in images_files:
                predictions4, probabilities4 = prediction3.predictImage(os.path.join(execution_path, image_file), result_count=result_count )
                print(f"\n\nThe DenseNet model's top {result_count} predictions with probability for {image_file}: ")
                for eachprediction4, eachProbability4 in zip(predictions4, probabilities4):
                    print(eachprediction4 , " : " , eachProbability4)
        except Exception as e:
            print("\n\nDenseNet didn't work.")
            print(e)
            print("Next model..")
    else:
        print('no models worked...')
