This page describes how to train the Inception Resnet v1 model using triplet loss. It should however be mentioned that training using triplet loss is trickier than training using softmax. But when the training set contains a significant amount of classes (more than 100 000) the final layer and the softmax itself can become prohibitively large and then training using triplet loss can still work fine. It should be noted that this guide by no means contains the final recipe for how to train a model using triplet loss but should rather be considered a work in progress.

To train a model with better performance, please refer to [Classifier training of Inception-ResNet-v1](Classifier-training-of-inception-resnet-v1.md).

## 1. Install Tensorflow
The current version of this FaceNet implementation requires Tensorflow version r1.0. It can be installed using [pip](https://www.tensorflow.org/get_started/os_setup#pip_installation) or from [sources](https://www.tensorflow.org/get_started/os_setup#installing_from_sources).<br>
Since training of deep neural networks is extremely computationally intensive it is recommended to use a CUDA enabled GPU. The Tensorflow installation page has a detailed description of how to install CUDA as well.

## 2. Clone the FaceNet [repo](https://github.com/davidsandberg/facenet.git)
This is done using the command <br>
`git clone https://github.com/davidsandberg/facenet.git`

## 3. Set the python paths
Set the environment variable `PYTHONPATH` to point to the `src` directory of the cloned repo. This is typically done something like this<br>
`export PYTHONPATH=[...]/facenet/src`<br>
where `[...]` should be replaced with the directory where the cloned facenet repo resides.

## 4. Prepare training dataset(s)
### Dataset structure
It is assumed that the training dataset is arranged as below, i.e. where each class is a sub-directory containing the training examples belonging to that class.

    Aaron_Eckhart
        Aaron_Eckhart_0001.jpg

    Aaron_Guiel
        Aaron_Guiel_0001.jpg

    Aaron_Patterson
        Aaron_Patterson_0001.jpg

    Aaron_Peirsol
        Aaron_Peirsol_0001.jpg
        Aaron_Peirsol_0002.jpg
        Aaron_Peirsol_0003.jpg
        Aaron_Peirsol_0004.jpg
        ...

### Face alignment
For face alignment it is recommended to use [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment) which has been proven to give very good performance for alignment of train/test sets. The authors have been kind enough to provide an implementation of MTCNN based on Matlab and Caffe. In addition, a matlab script to align a dataset using this implementation can be found [here](https://github.com/davidsandberg/facenet/blob/master/tmp/align_dataset.m).

To simplify the usage of this project a python/tensorflow implementation of MTCNN is [provided](https://github.com/davidsandberg/facenet/tree/master/src/align). This implementation does not have any other external dependencies than Tensorflow and the runtime on LFW is similar to the matlab implementation.

`python src/align/align_dataset_mtcnn.py ~/datasets/casia/CASIA-maxpy-clean/  ~/datasets/casia/casia_maxpy_mtcnnpy_182 --image_size 182 --margin 44`

The face thumbnails generated by the above command are 182x182 pixels. The input to the Inception-ResNet-v1 model is 160x160 pixels giving some margin to use a random crop.
For the experiments that has been performed with the Inception-ResNet-v1 model an margin additional margin of 32 pixels has been used. The reason for this additional widen the bounding box given by the face alignment and give the CNN some additional contextual information. However, the setting of this parameter has not yet been studied and it could very well be that other margins results in better performance.

To speed up the alignment process the above command can be run in multiple processes. Below, the same command is ran using 4 processes. To limit the memory usage of each Tensorflow  session the parameter `gpu_memory_fraction` is set to 0.25, meaning that each session is allowed to use maximum 25% of the total GPU memory. Try to decrease the number of parallel process and increase the fraction of GPU memory for each session if the below command causes the GPU memory to run out of memory.
`for N in {1..4}; do python src/align/align_dataset_mtcnn.py ~/datasets/casia/CASIA-maxpy-clean/  ~/datasets/casia/casia_maxpy_mtcnnpy_182 --image_size 182 --margin 44 --random_order --gpu_memory_fraction 0.25 & done`

## 4. Start training
Training is started by running `train_tripletloss.py`. <br>
`python src/train_tripletloss.py --logs_base_dir ~/logs/facenet/ --models_base_dir ~/models/facenet/ --data_dir ~/datasets/casia/casia_maxpy_mtcnnalign_182_160 --image_size 160 --model_def models.inception_resnet_v1 --lfw_dir ~/datasets/lfw/lfw_mtcnnalign_160 --optimizer RMSPROP --learning_rate 0.01 --weight_decay 1e-4 --max_nrof_epochs 500`

When training is started subdirectories for training session named after the data/time training was started on the format `yyyymmdd-hhmm` is created in the directories `log_base_dir` and `models_base_dir`. The parameter `data_dir` is used to point out the location of the training dataset. It should be noted that the union of several datasets can be used by separating the paths with a colon. Finally, the descriptor of the inference network is given by the `model_def` parameter. In the example above, `models.inception_resnet_v1` points to the `inception_resnet_v1` module in the package `models`. This module must define a function `inference(images, ...)`, where `images` is a placeholder for the input images (dimensions <?,160,160,3>) and returns a reference to the `embeddings` variable.

If the parameter `lfw_dir` is set to point to a the base directory of the LFW dataset the model is evaluated on LFW once every 1000 batches. For information on how to evaluate an existing model on LFW, please refer to the [Validate-on-LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-LFW) page. If no evaluation on LFW is desired during training it is fine to leave the `lfw_dir` parameter empty. However, please note that the LFW dataset that is used here should have been aligned in the same way as the training dataset.  

## 5. Running TensorBoard
While FaceNet training is running it can be interesting to monitor the learning process. This can be done using [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard). To start TensorBoard, run the command <br>`tensorboard --logdir=~/logs/facenet --port 6006`<br> and then point your web browser to <br>http://localhost:6006/