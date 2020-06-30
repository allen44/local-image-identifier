
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

#Initialize the Model
from imageai.Prediction import ImagePrediction
result_count = 5
prediction = ImagePrediction()

prediction.setModelTypeAsSqueezeNet()
prediction.setModelPath(os.path.join(execution_path, "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel()
predictions, probabilities = prediction.predictImage(os.path.join(execution_path, image_filename), result_count=result_count )
print(f"The SqueezeNet model's top {result_count} predictions with probability: ")
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)

prediction2 = ImagePrediction()
prediction2.setModelTypeAsResNet()
prediction2.setModelPath(os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
prediction2.loadModel()
predictions2, probabilities2 = prediction2.predictImage(os.path.join(execution_path, image_filename), result_count=result_count )
print(f"The ResNet model's top {result_count} predictions with probability: ")
for eachPrediction2, eachProbability2 in zip(predictions2, probabilities2):
    print(eachPrediction2 , " : " , eachProbability2)

prediction3 = ImagePrediction()
prediction3.setModelTypeAsInceptionV3()
prediction3.setModelPath(os.path.join(execution_path, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
prediction3.loadModel()
predictions3, probabilities3 = prediction3.predictImage(os.path.join(execution_path, image_filename), result_count=result_count )
print(f"The Inception model's top {result_count} predictions with probability: ")
for eachprediction3, eachProbability3 in zip(predictions3, probabilities3):
    print(eachprediction3 , " : " , eachProbability3)

# prediction4  = ImagePrediction()
# prediction4.setModelTypeAsDenseNet()
# prediction4.setModelPath(os.path.join(execution_path, "DenseNet-BC-121-32.h5"))
# prediction4.loadModel()
# predictions4, probabilities4 = prediction3.predictImage(os.path.join(execution_path, image_filename), result_count=result_count )
# print(f"The DenseNet model's top {result_count} predictions with probability: ")
# for eachprediction4, eachProbability4 in zip(predictions4, probabilities4):
#     print(eachprediction4 , " : " , eachProbability4)