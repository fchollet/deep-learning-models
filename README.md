# Trained image classification models for Keras

**THIS REPOSITORY IS DEPRECATED. USE THE MODULE `keras.applications` INSTEAD.**

Pull requests will not be reviewed nor merged. Direct any PRs to `keras.applications`. Issues are not monitored either.

----

This repository contains code for the following Keras models:

- VGG16
- VGG19
- ResNet50
- Inception v3
- CRNN for music tagging

All architectures are compatible with both TensorFlow and Theano, and upon instantiation the models will be built according to the image dimension ordering set in your Keras configuration file at `~/.keras/keras.json`. For instance, if you have set `image_dim_ordering=tf`, then any model loaded from this repository will get built according to the TensorFlow dimension ordering convention, "Width-Height-Depth".

Pre-trained weights can be automatically loaded upon instantiation (`weights='imagenet'` argument in model constructor for all image models, `weights='msd'` for the music tagging model). Weights are automatically downloaded if necessary, and cached locally in `~/.keras/models/`.

## Examples

### Classify images

```python
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
# print: [[u'n02504458', u'African_elephant']]
```

### Extract features from images

```python
from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```

### Extract features from an arbitrary intermediate layer

```python
from vgg19 import VGG19
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model

base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
```

## References

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) - please cite this paper if you use the VGG models in your work.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - please cite this paper if you use the ResNet model in your work.
- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567) - please cite this paper if you use the Inception v3 model in your work.
- [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)

Additionally, don't forget to [cite Keras](https://keras.io/getting-started/faq/#how-should-i-cite-keras) if you use these models.


## License

- All code in this repository is under the MIT license as specified by the LICENSE file.
- The ResNet50 weights are ported from the ones [released by Kaiming He](https://github.com/KaimingHe/deep-residual-networks) under the [MIT license](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE).
- The VGG16 and VGG19 weights are ported from the ones [released by VGG at Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) under the [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).
- The Inception v3 weights are trained by ourselves and are released under the MIT license.
