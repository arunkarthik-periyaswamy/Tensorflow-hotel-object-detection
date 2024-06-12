<!--Object Detection Code Using TensorFlow Object Detection API
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
-->

# Advanced Object Detection for Hotel Items using TensorFlow
![Python Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Python_logo_and_wordmark.svg/400px-Python_logo_and_wordmark.svg.png)
![TensorFlow Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/TensorFlow_logo.svg/400px-TensorFlow_logo.svg.png)
![TFLite Logo](https://storage.googleapis.com/gweb-developer-goog-blog-assets/images_archive/original_images/image1_v7xhr8h.png)


## Abstract

Object detection is extensively applied in various domains such as vehicle detection, face recognition, autonomous driving, and pedestrian monitoring. TensorFlow's Object Detection API is a robust tool that empowers users to rapidly develop and deploy sophisticated image recognition applications. Object detection encompasses not only classifying and recognizing objects within an image but also localizing these objects and delineating them with bounding boxes. This project primarily focuses on configuring the environment and utilizing TensorFlow Lite (TFLite) models for detecting hotel items such as jugs, cups, and flasks. The TensorFlow Object Detection API has been leveraged for model training, specifically employing the Single Shot Multibox Detector (SSD) with MobileNet V2 architecture.

## Usage

### Training the Model
To train the object detection model, execute the following command:

```bash
python train.py
```

### Deploying the Model
To deploy the trained model, execute the following command:

```bash
python deploy.py
```

## Prerequisites

- **TensorFlow**: A comprehensive open-source platform for machine learning.
- **TensorFlow Object Detection API**: A library for training and deploying object detection models.

### Installation
To install the TensorFlow Object Detection API, follow the instructions provided in the [TensorFlow Object Detection API documentation](https://github.com/tensorflow/models/tree/master/research/object_detection).

## Training Process

To train the model, a pipeline configuration file must be created. This file defines the model architecture, training parameters, and input/output configurations.

With the pipeline configuration file in place, initiate the training process using the `train.py` script. The script accepts the following parameters:

- **master**: The TensorFlow master server name.
- **task**: The task ID.
- **num_clones**: Number of model replicas per worker.
- **clone_on_cpu**: Deploy clones on CPU.
- **worker_replicas**: Number of worker replicas.
- **ps_tasks**: Number of parameter server tasks.
- **train_dir**: Directory for saving checkpoints and training summaries.
- **pipeline_config_path**: Path to the `TrainEvalPipelineConfig` configuration file.
- **train_config_path**: Path to the `TrainConfig` configuration file.
- **input_config_path**: Path to the `InputReader` configuration file.
- **model_config_path**: Path to the `DetectionModel` configuration file.

## Deployment Process

To deploy the model, use the `deploy.py` script with the following parameters:

- **model_path**: Path to the trained model.
- **image_path**: Path to the image for detection.
- **output_path**: Path to save the output image with detected objects.

## Example

### Training the Model
```bash
python train.py
```

### Deploying the Model
```bash
python deploy.py --model_path trained_model.pb --image_path image.jpg --output_path output.jpg
```
![Hotel Object Detection Output](https://github.com/arunkarthik-periyaswamy/Tensorflow-hotel-object-detection/blob/master/hotel_testing_output.png?raw=true)

*Hotel Object Detection Output*

## Troubleshooting
For any issues encountered during training or deployment, refer to the [TensorFlow Object Detection API documentation](https://github.com/tensorflow/models/tree/master/research/object_detection).

## Technologies Used

- **Python**: ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
- **TensorFlow**: ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
- **TensorFlow Object Detection API**: ![TensorFlow](https://img.shields.io/badge/TensorFlow-API-orange)
- **TensorFlow Lite (TFLite)**: ![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-TFLite-blue)
- **Single Shot Multibox Detector (SSD)**: ![SSD](https://img.shields.io/badge/SSD-Algorithm-green)
- **MobileNet V2**: ![MobileNet V2](https://img.shields.io/badge/MobileNet%20V2-Neural%20Network-brightgreen)



---

This document provides a comprehensive guide to setting up, training, and deploying an object detection model using TensorFlow's Object Detection API. It includes all necessary steps and commands to ensure a smooth implementation of advanced object detection for specific hotel items.
