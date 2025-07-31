# Computer-Vision
Exercises of Computer Vision with Deep Learning

```
|-- CIFAR10/
|   |-- Wide Resnet/
|   |-- Densenet/
|
|-- Gender Recognision/
|
|-- Car Model Identification/
```

## Basic implementations on CIFAR10

<img width="330" height="235" alt="image" src="https://github.com/user-attachments/assets/5d0df958-60ba-40dd-b464-635d605280d7" />

**Goals:**

* Implement advanced topologies: Wide Resnet and Dense Net.
* Implement different data augmentation

## Gender Recognision
Images from "Labeled Faces in the Wild" dataset (LFW) in realistic scenarios, poses and gestures.

<img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/e72f599d-1403-42a7-a1e1-c1d0bb11a12b" />

**Goals:**
* Implement a model with >98% accuracy over test set
* Implement a model with >95% accuracy with less than 100K parameters

## Car Model identification with bi-linear models
Images of 20 different models of cars.

**Goals:**
  * Load pre-trained VGG16, Resnet... models
  * Connect this pre-trained models and form a bi-linear
  * Connect models with operations (outproduct)
  * Train freezing weights first, unfreeze after some epochs, very low learning rate
  * Implement a model with >65% accuracy

<img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/ddc72396-3b4a-41ff-a8b3-42ecaba9ad84" />


# Results


| Problem                          | Accuracy (%) |
|----------------------------------|--------------|
| WideResNet (CIFAR10)             | 93.75        |
| WideResNet (CIFAR10)             | 95.67        |
| DenseNet-121-BC (CIFAR10)        | 91.59        |
| DenseNet-201-BC (CIFAR10)        | 92.60        |
| Gender Recognition (<1M params)  | 96.19        |
| Gender Recognition (>98%)        | 98.26        |
| Car Model Identification         | 71.94        |
