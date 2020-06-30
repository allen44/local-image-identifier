
#Define the execution path
import os
execution_path = os.getcwd()
print(execution_path)


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
image_files = []
for eachPicture in allfiles:
    if eachPicture.endswith(".png") or eachPicture.endswith(".jpg"):
        image_files.append(eachPicture)
print(image_files)

#Initialize the Model
from imageai.Prediction import ImagePrediction
result_count = 5

#Initializxe a text file to write to
def write_to_txt(line):
    with open('results.txt', mode='a') as output:
        output.write(line)

for model_abs_path in models_abs_paths:
    multiple_prediction = ImagePrediction()
    if 'squeezenet' in model_abs_path.lower():
        try:
            multiple_prediction.setModelTypeAsSqueezeNet()
            multiple_prediction.setModelPath(model_abs_path)
            multiple_prediction.loadModel()
            results_array = multiple_prediction.predictMultipleImages(image_files, result_count_per_image=result_count)
            # print(f"results_array: {results_array}")
            results_array_idx = 0
            for each_result in results_array:
                predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
                line = (f"\n\nThe SqueezeNet model's top {result_count} predictions with probability for {image_files[results_array_idx]}: ")
                print(line, end='')
                write_to_txt(line)
                for index in range(len(predictions)):
                    line = f"\n{predictions[index]}: {percentage_probabilities[index]}"
                    print(line, end='')
                    write_to_txt(line)
                print("-----------------------")
                results_array_idx = results_array_idx + 1
        except Exception as e:
            print("\n\nSqueezeNet didn't work.")
            print(e)
            print("Next model..")
    elif 'resnet' in model_abs_path.lower():
        try:    
            multiple_prediction.setModelTypeAsResNet()
            multiple_prediction.setModelPath(model_abs_path)
            multiple_prediction.loadModel()
            results_array = multiple_prediction.predictMultipleImages(image_files, result_count_per_image=result_count)
            # print(f"results_array: {results_array}")
            results_array_idx = 0
            print(f"\n\nThe ResNet model's top {result_count} predictions with probability for each image:")
            for each_result in results_array:
                predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
                line = (f"\n\nThe ResNet model's top {result_count} predictions with probability for {image_files[results_array_idx]}: ")
                print(line, end='')
                write_to_txt(line)
                for index in range(len(predictions)):
                    line = f"\n{predictions[index]}: {percentage_probabilities[index]}"
                    print(line, end='')
                    write_to_txt(line)
                print("-----------------------")
                results_array_idx = results_array_idx + 1
        except Exception as e:
            print("\n\n\ResNet didn't work.")
            print(e)
            print("Next model..")
    elif 'inception' in model_abs_path.lower():
        try:
            multiple_prediction.setModelTypeAsInceptionV3()
            multiple_prediction.setModelPath(model_abs_path)
            multiple_prediction.loadModel()
            results_array = multiple_prediction.predictMultipleImages(image_files, result_count_per_image=result_count)
            # print(f"results_array: {results_array}")
            results_array_idx = 0
            for each_result in results_array:
                predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
                line = (f"\n\nThe Inception model's top {result_count} predictions with probability for {image_files[results_array_idx]}: ")
                print(line, end='')
                write_to_txt(line)
                for index in range(len(predictions)):
                    line = f"\n{predictions[index]}: {percentage_probabilities[index]}"
                    print(line, end='')
                    write_to_txt(line)
                print("-----------------------")
                results_array_idx = results_array_idx + 1
        except Exception as e:
            print("\n\ninception didn't work.")
            print(e)
            print("Next model..")
    # # elif 'densenet' in model_abs_path.lower():
    #     try:
    #         multiple_prediction.setModelTypeAsDenseNet()
    #         multiple_prediction.setModelPath(model_abs_path)
    #         multiple_prediction.loadModel()
    #         results_array = multiple_prediction.predictMultipleImages(image_files, result_count_per_image=result_count)
    #         # print(f"results_array: {results_array}")
    #         results_array_idx = 0
    #         for each_result in results_array:
    #             predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
    #             line = (f"\n\nThe SqueezeNet model's top {result_count} predictions with probability for {image_files[results_array_idx]}: ")
    #             print(line, end='')
    #             write_to_txt(line)
    #             for index in range(len(predictions)):
                    # line = f"\n{predictions[index]}: {percentage_probabilities[index]}"
                    # print(line, end='')
                    # write_to_txt(line)
    #             print("-----------------------")
    #             results_array_idx = results_array_idx + 1
    #     except Exception as e:
    #         print("\n\nDenseNet didn't work.")
    #         print(e)
    #         print("Next model..")
    else:
        print('no models worked...')
