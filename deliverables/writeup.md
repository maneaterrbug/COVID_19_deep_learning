# COVID-19 and Convolutional Neural Nets
Matt Ryan

## Abstract
---


## Design
---

---
## Data

The data used in this project consists of 746 CT-scans of lungs in the form of png files, 349 of which were representative of clinical findings of COVID-19 (the scans were taken from a total of 216 patients) and the remainder from COVID-19 negative lungs. The images were not standard sized, and despite appearing grey-scale were 3-channel. The image data was collected and pre-processed by a team of researchers from University of California San Diego and University of California Berkley; the data can be found on a github repository [here](https://github.com/UCSD-AI4H/COVID-CT), and the team's discussion of collection and processing methodologies can be found in [this paper](https://arxiv.org/pdf/2003.13865.pdf).

The image data was downloaded from the repository to a local directory with the COVID-positive and COVID-negative images separated into separate folders. From there, the files were shuffled and sorted into separate training, validation, and testing directories along a 70-15-15 split, with corresponding COVID-positive and COVID-negative sub-directories. 

## Algorithms

### *Pre-processing*

For the base model using random forest modeling, the image data was resized into (256, 256, 3) images then loaded into train, val and test numpy arrays. Each image in each set array was flattened before being fed into a PCA fit and transform, allowing the random forest model to be trained on and to predict on the transformed image data.

For the baseline and transfer learning convolutional neural networks, the images were loaded into train, val, and test BatchDataset objects using *tensorflow.keras.preprocessing.image.image_dataset_from_dictionary()*. The *image_dataset_from_dictionary()* method by default resizes images to (256,256,3).

### *Building base models*

I developed two baseline models for benchmarking. For the first, I built a random forest classifer with 100 estimators and trained on the test data. This model predicted on the validation data with an accuracy of 0.519, negligibly better than a random binary guesser would perform. 

To see how a convolutional NN model might perform on image data of this kind, I built a custom baseline convolutional NN using the Keras Sequential model. This model was composed of 3 convolutional blocks, each with 2 Conv2d layers (with number of filters equal to 16, 32, and 64 for each respective block, a kernel size of 3, and relu activation),2 40% dropout layers, and a pooling layer. Finally the output of the convolutional blocks was fed into a fully connected block with 1 flattening layer and 2 20 node dense layers before being output from a 1 node output layer using sigmoid activation.

![]('../resources/nn_layers.png')

Despite the model's relative depth, it proved computationally costly to train, and was not feasible to train on a high number of epochs. Ultimately, this model performed best around 30 epochs of training, and predicted COVID-19 presence on the validation set with 63.9% accuracy.

### *Applying transfer learning*

Because the results of the baseline CNN showed promising gains in accuracy, but diminishing returns in accuracy with respect to time complexity, I decided to apply transfer learning. Using the VGG16 as a base hidden layer, all layers within the VGG16 layer were frozen and a fully connected block composed of a single flatten layer and 3 dense layers (of 1000, 500, and 250 nodes, respectively) were appended. To prevent any overfitting on our small dataset, 20% dropout layers were fit between the dense layers, the momentum and learning rate of the SGD optimizer were set high, and an early-stopping callback was applied. In training, the early-stopping callback interupted the model at epoch 7, achieving a peak predictive accuracy on the val set of 79.13%.

## Tools

*Image loading and pre-processing*
- cv2
- NumPy
- SKLearn PCA
- Keras preprocessing.image

*Base Modelling*
- SKLearn RandomForestClassifier
- Tensorflow Keras Sequential
- Tensorflow Keras Layers

*Transfer Learning*
- VGG16

## Communication

Despite initial CNN modeling proving 



Future work: 
- preprocessing
  - image augmentation
  - image sizing/standardizing
  - bringing in additional scans


@article{zhao2020COVID-CT-Dataset,
  title={COVID-CT-Dataset: a CT scan dataset about COVID-19},
  author={Zhao, Jinyu and Zhang, Yichen and He, Xuehai and Xie, Pengtao},
  journal={arXiv preprint arXiv:2003.13865}, 
  year={2020}
}