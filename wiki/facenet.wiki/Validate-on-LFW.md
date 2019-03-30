## 1. Install dependencies
In the below description it is assumed that<br>
a) Tensorflow has been [installed](Running-training#1-install-tensorflow), and<br>
b) the Facenet [repo](https://github.com/davidsandberg/facenet.git) has been cloned, and<br>
c) the [required python modules](https://github.com/davidsandberg/facenet/blob/master/requirements.txt) has been installed.

## 2. Download the LFW dataset
#### 1. Download unaligned images from [here](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
In this example the archive is downloaded to `~/Downloads`.
#### 2. Extract the unaligned images to local storage
Assuming you have a directory `~/datasets` for storing datasets
```
cd ~/datasets
mkdir -p lfw/raw
tar xvf ~/Downloads/lfw.tgz -C lfw/raw --strip-components=1
```

## 3. Set the python path
Set the environment variable `PYTHONPATH` to point to the `src` directory of the cloned repo. This is typically done something like this<br>
`export PYTHONPATH=[...]/facenet/src`<br>
where `[...]` should be replaced with the directory where the cloned facenet repo resides.

## 4. Align the LFW dataset
Alignment of the LFW dataset can be done using `align_dataset_mtcnn` in the `align` module.

Alignment of the LFW dataset is done something like this:<br>
```
for N in {1..4}; do \
python src/align/align_dataset_mtcnn.py \
~/datasets/lfw/raw \
~/datasets/lfw/lfw_mtcnnpy_160 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.25 \
& done
```

The parameter `margin` controls how much wider aligned image should be cropped compared to the bounding box given by the face detector. 32 pixels with an image size of 160 pixels corresponds to a margin of 44 pixels with an image size of 182, which is the image size that has been used for training of the model below.

## 5. Download pre-trained model (optional)
If you don not have your own trained model that you would like to test and easy way forward is to download a pre-trained model to run the test on.
One such model can be found [here](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-). Download and extract the model and place in your favorite models directory (in this example we use `~/models/facenet/`). After extracting the archive there should be a new folder `20180402-114759` with the contents<br>
```
20180402-114759.pb
model-20180402-114759.ckpt-275.data-00000-of-00001
model-20180402-114759.ckpt-275.index
model-20180402-114759.meta

```

## 6. Run the test
The test is ran using `validate_on_lfw`:<br>
```
python src/validate_on_lfw.py \
~/datasets/lfw/lfw_mtcnnpy_160 \
~/models/facenet/20180402-114759 \
--distance_metric 1 \
--use_flipped_images \
--subtract_mean \
--use_fixed_image_standardization
```

This will <br>
a) load the model,<br>
b) load and parse the text file with the image pairs, <br>
c) calculate the embeddings for all the images (as well as their horizontally flipped versions) in the test set,<br>
d) calculate the accuracy, validation rate (@FAR=-10e-3), the Area Under Curve (AUC) and the Equal Error Rate (EER) performance measures.

A typical output from the the test looks like this:
```
Model directory: /home/david/models/20180402-114759/
Metagraph file: model-20180402-114759.meta
Checkpoint file: model-20180402-114759.ckpt-275
Runnning forward pass on LFW images
........................
Accuracy: 0.99650+-0.00252
Validation rate: 0.98367+-0.00948 @ FAR=0.00100
Area Under Curve (AUC): 1.000
Equal Error Rate (EER): 0.004
```
