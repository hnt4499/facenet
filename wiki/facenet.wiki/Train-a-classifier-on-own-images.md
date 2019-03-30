This page describes how to train your own classifier on your own dataset. Here it is assumed that you have followed e.g. the guide [Validate on LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw) to install dependencies, clone the FaceNet repo, set the python path etc and aligned the LFW dataset (at least for the LFW experiment). In the examples below the frozen model `20170216-091149` is used. Using a frozen graph significantly speeds up the loading of the model.

# Train a classifier on LFW
For this experiment we train a classifier using a subset of the LFW images. The LFW dataset is split into a training and a test set. Then a pretrained model is loaded, and this model is then used to generate features for the selected images. The pretrained model is typically trained on a much larger dataset in order to give decent performance (in this case a subset of the MS-Celeb-1M dataset).

* Split the dataset into train and test sets
* Load a pretrained model for feature extraction
* Calculate embeddings for images in the dataset
* mode=TRAIN:
    * Train the classifier using embeddings from the train part of a dataset
    * Save the trained classification model as a python pickle
* mode=CLASSIFY:
    * Load a classification model
    * Test the classifier using embeddings from the test part of a dataset

#### Training a classifier on the training set part of the dataset is done as:
`python src/classifier.py TRAIN /home/david/datasets/lfw/lfw_mtcnnalign_160 /home/david/models/model-20170216-091149.pb ~/models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset`

The output from the training is shown below:
```
Number of classes: 19
Number of images: 665
Loading feature extraction model
Model filename: /home/david/models/model-20170216-091149.pb
Calculating features for images
Training classifier
Saved classifier model to file "/home/david/models/lfw_classifier.pkl"
```

#### The trained classifier can later be used for classification using the test set:

`python src/classifier.py CLASSIFY ~/datasets/lfw/lfw_mtcnnalign_160 ~/models/model-20170216-091149.pb ~/models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset`

Here the test set part of the dataset is used for classification and the classification result together with the classification probability is shown. The classification accuracy for this subset is ~0.98.

```
Number of classes: 19
Number of images: 1202
Loading feature extraction model
Model filename: /home/david/models/export/model-20170216-091149.pb
Calculating features for images
Testing classifier
Loaded classifier model from file "/home/david/lfw_classifier.pkl"
   0  Ariel Sharon: 0.583
   1  Ariel Sharon: 0.611
   2  Ariel Sharon: 0.670
...
...
...
1198  Vladimir Putin: 0.588
1199  Vladimir Putin: 0.623
1200  Vladimir Putin: 0.566
1201  Vladimir Putin: 0.651
Accuracy: 0.978
```

# Train a classifier on your own dataset
So maybe you want to automatically categorize your private photo collection. Or you have a security camera that you want to automatically recognize the members of your family. Then it's likely that you would like to train a classifier on your own dataset. In this case `classifier.py` program can be used also for this. I have created my own train and test datasets by copying subsets of the LFW datasets. In this example the 5 first images of each class was used for training and the next 5 images was used for testing.

The classes that was used are:
* Ariel_Sharon
* Arnold_Schwarzenegger
* Colin_Powell
* Donald_Rumsfeld
* George_W_Bush
* Gerhard_Schroeder
* Hugo_Chavez
* Jacques_Chirac
* Tony_Blair
* Vladimir_Putin


#### The training of the classifier is done in a similar way as before:

`python src/classifier.py TRAIN ~/datasets/my_dataset/train/ ~/models/model-20170216-091149.pb ~/models/my_classifier.pkl --batch_size 1000`

The training of the classifier takes a few seconds (after loading the pre-trained model) and the output is shown below. Since this is a very simple dataset the accuracy is very good.
```
Number of classes: 10
Number of images: 50
Loading feature extraction model
Model filename: /home/david/models/model-20170216-091149.pb
Calculating features for images
Training classifier
Saved classifier model to file "/home/david/models/my_classifier.pkl"
```

This is how the directory containing the test set is organized:
```
/home/david/datasets/my_dataset/test
├── Ariel_Sharon
│   ├── Ariel_Sharon_0006.png
│   ├── Ariel_Sharon_0007.png
│   ├── Ariel_Sharon_0008.png
│   ├── Ariel_Sharon_0009.png
│   └── Ariel_Sharon_0010.png
├── Arnold_Schwarzenegger
│   ├── Arnold_Schwarzenegger_0006.png
│   ├── Arnold_Schwarzenegger_0007.png
│   ├── Arnold_Schwarzenegger_0008.png
│   ├── Arnold_Schwarzenegger_0009.png
│   └── Arnold_Schwarzenegger_0010.png
├── Colin_Powell
│   ├── Colin_Powell_0006.png
│   ├── Colin_Powell_0007.png
...
...
```

#### Classification on the test set can be ran using:

`python src/classifier.py CLASSIFY ~/datasets/my_dataset/test/ ~/models/model-20170216-091149.pb ~/models/my_classifier.pkl --batch_size 1000`

```
Number of classes: 10
Number of images: 50
Loading feature extraction model
Model filename: /home/david/models/model-20170216-091149.pb
Calculating features for images
Testing classifier
Loaded classifier model from file "/home/david/models/my_classifier.pkl"
   0  Ariel Sharon: 0.452
   1  Ariel Sharon: 0.376
   2  Ariel Sharon: 0.426
...
...
...
  47  Vladimir Putin: 0.418
  48  Vladimir Putin: 0.453
  49  Vladimir Putin: 0.378
Accuracy: 1.000
```

This code is aimed to give some inspiration and ideas for how to use the face recognizer, but it is by no means a useful application by itself.
Some additional things that could be needed for a real life application include:

* Include face detection in a face detection and classification pipe line
* Use a threshold for the classification probability to find unknown people instead of just using the class with the highest probability
