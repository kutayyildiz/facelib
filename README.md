# facelib

Face recognition python library(tensorflow, opencv).

## Usage (console)

try `facelib --help` to discover more

### Train

***Console command***

```bash
foo@bar:~$ python3 -m facelib train train_images/ lotr
```

***Console out***

```bash
Current pipeline: ssd_int8_cpu, mobilenetv2_fp32_cpu, densenet_fp32_cpu
Classifier named `lotr` succesfully trained and saved.
```

***Folder structure:***  
train_images/  
├───elijah_wood/  
├───├──0.jpg  
├───├──1.jpg  
├───liv_tyler/  
├───├──0.jpg  
├───├──1.jpg  
...  

***Some of the train images:***  
https://github.com/kutayyildiz/facelib/raw/master/facelib/facelib/_demo/lotr_cast.png
| Image Name                      | Image                                                                                                                |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| train_images/ian_mckellen/0.jpg | ![ianmckellen](https://github.com/kutayyildiz/facelib/raw/master/facelib/_demo/lotr/train_images/ian_mckellen/0.jpg) |
| train_images/seanastin/0.jpg    | ![seanastin](https://github.com/kutayyildiz/facelib/raw/master/facelib/_demo/lotr/train_images/sean_astin/0.jpg)     |

### Predict

***Console command***

-c: crop  
-p: plot  
-clf: classifier

```bash
foo@bar:~$ python3 -m facelib predict test_images/ -clf lotr -c -p
```
***Console out***

```bash
Current pipeline: ssd_int8_cpu, mobilenetv2_fp32_cpu, densenet_fp32_cpu
1.jpg
├───10 faces detected
├───['billy_boyd', 'sean_astin', 'viggo_mortensen', 'elijah_wood', 'liv_tyler', 'dominic_monaghan', 'sean_bean', 'ian_mckellen', 'peter_jackson', 'orlando_bloom']
2.jpg
├───5 faces detected
├───['dominic_monaghan', 'billy_boyd', 'elijah_wood', 'sean_astin', 'peter_jackson']
3.jpg
├───6 faces detected
├───['orlando_bloom', 'dominic_monaghan', 'john_rhys_davies', 'sean_astin', 'elijah_wood', 'billy_boyd']
0.jpeg
├───5 faces detected
├───['dominic_monaghan', 'orlando_bloom', 'elijah_wood', 'liv_tyler', 'billy_boyd']
```

***Folder structure:***  
test_images/  
├──0.jpeg  
├──1.jpg  
├──2.jpg  
├──3.jpg  

***Generated folders/files:***  
test_images_facelib_cropped/  
├───elijah_wood/  
├───├──2_2.jpg  
├───├──3_1.jpg  
├───├──4_3.jpg  
├───liv_tyler/  
├───├──3_0.jpg  
├───├──4_1.jpg  
...

***Some of the generated images:***  
| Image Name                                      | Image                                                                                                                               |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| test_images_facelib_cropped/billy_boyd/0_1.jpg  | ![billyboyd](https://github.com/kutayyildiz/facelib/raw/master/facelib/_demo/lotr/test_images_facelib_cropped/billy_boyd/0_1.jpg)   |
| test_images_facelib_cropped/liv_tyler/4_1.jpg   | ![livtyler](https://github.com/kutayyildiz/facelib/raw/master/facelib/_demo/lotr/test_images_facelib_cropped/liv_tyler/4_1.jpg)     |
| test_images_facelib_cropped/elijah_wood/3_1.jpg | ![elijahwood](https://github.com/kutayyildiz/facelib/raw/master/facelib/_demo/lotr/test_images_facelib_cropped/elijah_wood/3_1.jpg) |
| test_images_facelib_plotted/1.jpg               | ![1](https://github.com/kutayyildiz/facelib/raw/master/facelib/_demo/lotr/test_images_facelib_plotted/1.jpg)                        |

## Usage (python)

```python
from facelib import facerec
import cv2
# You can use face_detector, landmark_detector or feature_extractor individually using .predict method. e.g.(bboxes = facedetector.predict(img))
face_detector = facerec.SSDFaceDetector()
landmark_detector = facerec.LandmarkDetector()
feature_extractor = facerec.FeatureExtractor()

pipeline = facerec.Pipeline(face_detector, landmark_detector, feature_extractor)
path_img = './path_to_some_image.jpg'
img = cv2.imread(path_img)
img = img[...,::-1] # cv2 returns bgr format but every method inside this package takes rgb format
bboxes, landmarks, features = pipeline.predict(img)
# Note that values returned (bboxes and landmarks) are in fraction.[0,1]
```

## Installation

### Pip installation

```bash
pip3 install facelib
```

### TFLite runtime installation

To use facelib.facerec package use the following bash command to install tflite-runtime pip package.

```bash
python3 -m facelib --install-tflite
```

or you can install from [tensorflow.org](https://www.tensorflow.org/lite/guide/python)

### Dev package

Tensorflow is required for facelib.dev package. If you wish you can download facelib with tensorflow using the following command.

```bash
pip3 install facelib[dev]
```

## Info

### Dataset

Feature extraction models are trained using insightfaces [MS1M-Arcface.](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)  
Landmark Detection models are trained using [VggFace2.](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)

## Contents

### Image Augmentation

- [x] Random augmentation for landmark detection

### Layers

- [x] DisturbLabel

### Face Alignment

- [x] Insightface
- [x] GoldenRatio
- [x] Custom Implementations

### TFRecords

- [ ] Widerface to TFRecords converter
- [ ] VggFace2 to TFRecords converter
- [ ] COFW to TFRecords converter

### Loss Functions

#### Feature Extraction

- [x] ArcFace
- [x] CombinedMargin
- [x] SphereFace(A-Softmax)
- [ ] Center
- [x] CosFace

#### Landmark Detection

- [x] EuclideanDistance(with different norms)

### Pretrained Models

#### Face Detection

- [x] SSD
- [ ] MTCNN

#### Face Feature Extraction

- [x] MobileFaceNet
- [x] SqueezeNet
- [x] MobileNet
- [x] MobileNetV2
- [x] DenseNet
- [x] NasNetMobile

#### Scripts

- [ ] Feature extraction model training
- [ ] Landmark detection model training
- [ ] Chokepoint test on pipeline

#### Facial Landmark Detection

- [ ] SqueezeNet
- [x] MobileNet
- [x] MobileNetV2
- [ ] DenseNet

## References

|                              |                                                                                                                                                                                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| WiderFace                    | Yang, Shuo, Ping Luo, Chen Change Loy, and Xiaoou Tang. “WIDER FACE: A Face Detection Benchmark.” ArXiv:1511.06523 [Cs], November 20, 2015. <https://arxiv.org/abs/1511.06523>                                                             |
| ArcFace                      | Deng, Jiankang, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. “ArcFace: Additive Angular Margin Loss for Deep Face Recognition.” ArXiv:1801.07698 [Cs], January 23, 2018. <https://arxiv.org/abs/1801.07698>                               |
| MobileFaceNet                | Chen, Sheng, Yang Liu, Xiang Gao, and Zhen Han. “MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices.” CoRR abs/1804.07573 (2018). <http://arxiv.org/abs/1804.07573>                                 |
| VggFace2                     | Cao, Qiong, Li Shen, Weidi Xie, Omkar M. Parkhi, and Andrew Zisserman. “VGGFace2: A Dataset for Recognising Faces across Pose and Age.” ArXiv:1710.08092 [Cs], October 23, 2017. <http://arxiv.org/abs/1710.08092>                         |
| DenseNet                     | G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely Connected Convolutional Networks,” arXiv:1608.06993 [cs], Jan. 2018. <http://arxiv.org/abs/1608.06993>                                                                 |
| GoldenRatio (face alignment) | M. Hassaballah, K. Murakami, and S. Ido, “Face detection evaluation: a new approach based on the golden ratio,” SIViP, vol. 7, no. 2, pp. 307–316, Mar. 2013. <http://link.springer.com/10.1007/s11760-011-0239-3>                         |
| SqueezeNet                   | F. N. Iandola, S. Han, M. W. Moskewicz, K. Ashraf, W. J. Dally, and K. Keutzer, “SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size,” arXiv:1602.07360 [cs], Feb. 2016.  <http://arxiv.org/abs/1602.07360> |
| MobileNet                    | A. G. Howard et al., “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,” arXiv:1704.04861 [cs], Apr. 2017. <http://arxiv.org/abs/1704.04861>                                                             |
| MobileNetV2                  | M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” arXiv:1801.04381 [cs], Jan. 2018. <http://arxiv.org/abs/1801.04381>                                                 |
| CosFace                      | H. Wang et al., “CosFace: Large Margin Cosine Loss for Deep Face Recognition,” arXiv:1801.09414 [cs], Jan. 2018. <http://arxiv.org/abs/1801.09414>                                                                                         |
| SphereFace                   | W. Liu, Y. Wen, Z. Yu, M. Li, B. Raj, and L. Song, “SphereFace: Deep Hypersphere Embedding for Face Recognition,” arXiv:1704.08063 [cs], Apr. 2017. <http://arxiv.org/abs/1704.08063>                                                      |
| Bottleneck Layer             | K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” arXiv:1512.03385 [cs], Dec. 2015. <http://arxiv.org/abs/1512.03385>                                                                                   |
| MS-Celeb-1M                  | Y. Guo, L. Zhang, Y. Hu, X. He, and J. Gao, “MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition,” arXiv:1607.08221 [cs], Jul. 2016. <http://arxiv.org/abs/1607.08221>                                                   |
| DisturbLabel                 | arXiv:1605.00055 [cs.CV]                                                                                                                                                                                                                   |
| Single Shot Detector         | [1]W. Liu et al., “SSD: Single Shot MultiBox Detector,” arXiv:1512.02325 [cs], Dec. 2016. <https://arxiv.org/abs/1512.02325>                                                                                                               |

## Links

|                        |                                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------------------- |
| Insightface            | <https://github.com/deepinsight/insightface>                                                              |
| Tensorflow             | <https://github.com/tensorflow/tensorflow>                                                                |
| Tensorflow-Addons      | <https://github.com/tensorflow/addons>                                                                    |
| Insightface-DatasetZoo | <https://github.com/deepinsight/insightface/wiki/Dataset-Zoo>                                             |
| Tensorflow-ModelZoo    | <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md> |
| Cascade Data           | <https://github.com/opencv/opencv/tree/master/data>                                                       |
| TFLite Python          | <https://www.tensorflow.org/lite/guide/python>                                                            |
