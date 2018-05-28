# MobilenetV1 Tensorflow to Chainer converter

Extracts the parameters of a MobilenetV1 model trained in Tensorflow for use in
Chainer. Follows the naming conventions seen [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

## Extracting
Run `pip install -r requirements.txt` and type `python convert.py 
/path/to/tgz/file`. For example:

```sh
python convert.py mobilenet_v1_1.0_224.tgz
```

This will return a file in the `chainer-models/` directory.

## Using the model

The converted model can be loaded with the serializer:

```python
import chainer
import mobilenet_v1_ch

model = mobilenet_v1_ch.MobilenetV1(depth_multiplier=1.0)
chainer.serializers.load_npz('chainer-models/mobilenet_v1_1.0_224.npz', model)
```

Make sure to change the depth multiplier accordingly.

The chainer model provided inherits from chainercv's `PickableSequentialChain`,
making it possible to return the output of intermediate layers:

```python
model.pick = ['conv2d_11', 'conv2d_13']
a1, a2 = model(img)
```

This will return the activations of the 11th and 13th depthwise separable
convolutional blocks.

