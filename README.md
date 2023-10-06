Object Detection Code Using TensorFlow Object Detection API
This code trains and deploys an object detection model using the TensorFlow Object Detection API. The model is trained to detect hotel items like a jug, cup, and flask.

Abstract

Object detection is widely utilized in several applications such as detecting vehicles, face detection, autonomous vehicles, and pedestrians on streets. TensorFlow's Object Detection API is a powerful tool that can quickly enable anyone to build and deploy powerful image recognition software. Object detection not only includes classifying and recognizing objects in an image but also localizes those objects and draws bounding boxes around them. This paper mostly focuses on setting up the environment and tflite model for detecting hotel items like jug, cup, and flask. We have used the Tensor Flow Object Detection API to train the model, and we have used the Single Shot Multibox Detector (SSD) MobileNet V2 algorithm for implementation.

Usage

To train the model, run the following command:

python train.py
To deploy the model, run the following command:

python deploy.py
Prerequisites

TensorFlow
TensorFlow Object Detection API
Installation

To install the TensorFlow Object Detection API, follow the instructions in the TensorFlow object detection API documentation: https://github.com/tensorflow/models/tree/master/research/object_detection.

Training

To train the model, you will need to create a pipeline configuration file. The pipeline configuration file specifies the model architecture, training parameters, and input and output configurations.

Once you have created a pipeline configuration file, you can train the model using the train.py script. The train.py script takes the following flags:

master: The name of the TensorFlow master to use.
task: The task id.
num_clones: The number of clones to deploy per worker.
clone_on_cpu: Whether to force clones to be deployed on CPU.
worker_replicas: The number of worker+trainer replicas.
ps_tasks: The number of parameter server tasks.
train_dir: The directory to save the checkpoints and training summaries.
pipeline_config_path: The path to a pipeline_pb2.TrainEvalPipelineConfig config file. If provided, other configs are ignored.
train_config_path: The path to a train_pb2.TrainConfig config file.
input_config_path: The path to an input_reader_pb2.InputReader config file.
model_config_path: The path to a model_pb2.DetectionModel config file.
Deployment

To deploy the model, you can use the deploy.py script. The deploy.py script takes the following flags:

model_path: The path to the trained model.
image_path: The path to the image to be detected.
output_path: The path to the output image with the detected objects.
Example

The following example shows how to train and deploy the model:

# Train the model.
python train.py

# Deploy the model.
python deploy.py --model_path trained_model.pb --image_path image.jpg --output_path output.jpg
Troubleshooting

If you are having problems training or deploying the model, please refer to the TensorFlow object detection API documentation: https://github.com/tensorflow/models/tree/master/research/object_detection.
