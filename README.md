
# Effective TensorFlow

Table of Contents
=================
1.  [TensorFlow Basics](#basics)
2.  [Understanding static and dynamic shapes](#shapes)
3.  [Scopes and when to use them](#scopes)
4.  [Broadcasting the good and the ugly](#broadcast)
5.  [Feeding data to TensorFlow](#data)
6.  [Take advantage of the overloaded operators](#overloaded_ops)
7.  [Understanding order of execution and control dependencies](#control_deps)
8.  [Control flow operations: conditionals and loops](#control_flow)
9.  [Prototyping kernels and advanced visualization with Python ops](#python_ops)
10. [Multi-GPU processing with data parallelism](#multi_gpu)
11. [Debugging TensorFlow models](#debug)
12. [Numerical stability in TensorFlow](#stable)
13. [Building a neural network training framework with learn API](#tf_learn)
14. [TensorFlow Cookbook](#cookbook)
    - [Get shape](#get_shape)
    - [Batch gather](#batch_gather)
    - [Beam search](#beam_search)
    - [Merge](#merge)
    - [Entropy](#entropy)
    - [KL-Divergence](#kld)
    - [Make parallel](#make_parallel)
    - [Leaky Relu](#leaky_relu)
    - [Batch normalization](#batch_norm)
    - [Squeeze and excitation](#squeeze_excite)
---

_We aim to gradually expand this series by adding new articles and keep the content up to date with the latest releases of TensorFlow API. If you have suggestions on how to improve this series or find the explanations ambiguous, feel free to create an issue, send patches, or reach out by email._

You can instantiate a notebook with the following examples in your browser using Binder (click the badge below):
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/MTDzi/EffectiveTensorflow/master?filepath=interactive_README.ipynb)
or create and environment:
```
conda env create -f environment.yml
```
and run the `"interactive_README.ipynb"` notebook on your local machine.

 _We encourage you to also check out the accompanied neural network training framework built on top of tf.contrib.learn API. The [framework](https://github.com/vahidk/TensorflowFramework) can be downloaded separately:_
```
git clone https://github.com/vahidk/TensorflowFramework.git
```

## TensorFlow Basics
<a name="basics"></a>

----

The most striking difference between TensorFlow and other numerical computation libraries such as NumPy is that operations in TensorFlow are symbolic. This is a powerful concept that allows TensorFlow to do all sort of things (e.g. automatic differentiation) that are not possible with imperative libraries such as NumPy. But it also comes at the cost of making it harder to grasp. Our attempt here is to demystify TensorFlow and provide some guidelines and best practices for more effective use of TensorFlow.

Let's start with a simple example, we want to multiply two random matrices. First we look at an implementation done in NumPy:


```python
import numpy as np

x = np.random.normal(size=[10, 10])
y = np.random.normal(size=[10, 10])
z = np.dot(x, y)

print(z)
```

    [[ 8.79409038e-01  3.81998372e+00  2.76142650e+00 -1.10443099e+00
       2.34207209e+00  1.09404800e+00 -4.16466436e+00 -1.31565590e+00
       3.93749701e+00  1.72239924e-02]
     [-1.72033492e+00  4.43523293e+00 -3.76654809e+00  5.92396749e+00
       3.75884227e+00 -2.21848562e+00 -8.01930215e+00  7.13999857e-02
       4.41499433e+00 -4.39070456e+00]
     [-9.51343146e-01  1.16684786e-02  4.28765724e+00 -3.99609470e+00
      -6.53963056e-01  1.66577398e+00  5.05442593e+00 -1.92111398e+00
       1.05661361e+00 -4.43813336e+00]
     [-1.94513644e+00  3.88850509e+00 -1.69990917e+00  4.46609852e+00
       1.97062190e+00 -4.94135502e+00 -3.55929104e+00 -8.71903366e-01
      -3.67583174e+00 -5.21093268e-01]
     [ 1.85089272e-01 -1.71569188e-01 -3.79389824e+00 -4.37158036e+00
       4.87769003e+00 -4.55368497e+00 -6.48179381e+00 -4.86127159e+00
      -2.33706956e-03 -5.53081260e+00]
     [ 2.86192318e+00 -7.16613203e+00  2.77425206e+00 -4.51729599e+00
       6.26984028e-01 -1.42636517e+00  6.12893619e+00 -3.29631539e+00
       5.23484105e+00  4.22569149e+00]
     [ 4.35239865e+00 -5.72362565e+00  7.37497678e-01 -3.76655609e+00
      -1.87693965e+00 -2.11167266e+00  8.41343975e+00  1.99359154e+00
      -4.37816850e+00  7.10725662e+00]
     [-6.59815696e+00  1.60422971e-01  1.13555917e+00  1.09342262e+00
       1.08585134e+00 -1.46826422e+00  1.41958562e+00 -3.69412211e+00
       4.88592707e-01 -8.40873143e+00]
     [ 4.06885256e+00 -3.47011533e+00 -3.03504023e+00  1.50729952e+00
      -3.86903644e+00  5.52951999e-01  4.49750807e+00  4.49436108e+00
      -4.05779149e+00  3.92145485e+00]
     [-2.03376992e-01 -5.27469628e+00 -1.33294216e+00  3.34499456e+00
      -1.46605132e+00  2.74644227e-01  6.51539515e+00  3.69561625e+00
       1.30654920e+00  1.77191815e+00]]


Now we perform the exact same computation this time in TensorFlow:


```python
import tensorflow as tf

x = tf.random_normal([10, 10])
y = tf.random_normal([10, 10])
z = tf.matmul(x, y)

sess = tf.Session()
z_val = sess.run(z)

print(z_val)
```

    [[-4.1316524   5.470432    0.59554493 -1.8662449   3.4217193  -1.3617333
      -1.7188292   1.5467906   0.67369133 -7.6193438 ]
     [-2.7073328   4.9765196   1.1192673   1.68204     0.7387767  -0.2509363
      -0.7454821  -1.802145    1.7182754  -2.9671009 ]
     [-1.1915479   0.52490574  4.980897    1.1170299   6.250311    5.714476
      -0.48571974 -3.3865023   3.1981876   4.770077  ]
     [-1.7634747   3.0931196   2.0398793  -3.5683167   2.6523361   3.5200636
      -2.777476    3.2244759   3.2137678  -2.200465  ]
     [ 2.2088833   2.5342038  -2.0832245   5.2913365   0.08185586 -3.002068
       0.84187114 -8.218136    0.33472735  0.67385757]
     [ 4.1537414   0.5846967  -2.5292733   3.1380126   0.0980214  -4.4586506
       0.52054334 -4.8179173   0.10286486 -0.3136202 ]
     [-1.3088549  -0.5607232  -1.8139035   0.1759216  -2.1102467  -0.7180903
       1.7647583  -1.9585884  -2.2129374  -2.015387  ]
     [-1.3420477  -1.5493006   0.6342718  -4.44202     0.5442125   2.4045124
      -0.8021609  -0.6007094  -0.95977414 -1.3432573 ]
     [ 2.5201898   1.0976803   0.9979652   2.3174958   3.1102614  -2.1573691
       2.0236979   2.9892766  -0.96625525  0.586673  ]
     [ 0.9586587  -0.62876415 -3.4533203   6.2302017   4.1884394  -3.1654315
      -0.49213636 -3.176754    2.2364793   2.6102982 ]]


Unlike NumPy that immediately performs the computation and produces the result, tensorflow only gives us a handle (of type Tensor) to a node in the graph that represents the result. If we try printing the value of z directly, we get something like this:


```python
z
```




    <tf.Tensor 'MatMul:0' shape=(10, 10) dtype=float32>



Since both the inputs have a fully defined shape, tensorflow is able to infer the shape of the tensor as well as its type. In order to compute the value of the tensor we need to create a session and evaluate it using Session.run() method.


***
__Tip__: When using Jupyter notebook make sure to call tf.reset_default_graph() at the beginning to clear the symbolic graph before defining new nodes.
***

To understand how powerful symbolic computation can be let's have a look at another example. Assume that we have samples from a curve (say f(x) = 5x^2 + 3) and we want to estimate f(x) based on these samples. We define a parametric function g(x, w) = w0 x^2 + w1 x + w2, which is a function of the input x and latent parameters w, our goal is then to find the latent parameters such that g(x, w) ≈ f(x). This can be done by minimizing the following loss function: L(w) = &sum; (f(x) - g(x, w))^2. Although there's a closed form solution for this simple problem, we opt to use a more general approach that can be applied to any arbitrary differentiable function, and that is using stochastic gradient descent. We simply compute the average gradient of L(w) with respect to w over a set of sample points and move in the opposite direction.


Here's how it can be done in TensorFlow:


```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Placeholders are used to feed values from python to TensorFlow ops. We define
# two placeholders, one for input feature x, and one for output y.
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Assuming we know that the desired function is a polynomial of 2nd degree, we
# allocate a vector of size 3 to hold the coefficients. The variable will be
# automatically initialized with random noise.
w = tf.get_variable("w", shape=[3, 1])

# We define yhat to be our estimate of y.
f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1)
yhat = tf.squeeze(tf.matmul(f, w), 1)

# The loss is defined to be the l2 distance between our estimate of y and its
# true value. We also added a shrinkage term, to ensure the resulting weights
# would be small.
loss = tf.nn.l2_loss(yhat - y) + 0.1 * tf.nn.l2_loss(w)

# We use the Adam optimizer with learning rate set to 0.1 to minimize the loss.
train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

def generate_data():
    x_val = np.random.uniform(-10.0, 10.0, size=100)
    y_val = 5 * np.square(x_val) + 3
    return x_val, y_val

sess = tf.Session()

losses_val = []
# Since we are using variables we first need to initialize them.
sess.run(tf.global_variables_initializer())
for _ in range(1000):
    x_val, y_val = generate_data()
    _, loss_val = sess.run([train_op, loss], {x: x_val, y: y_val})
    losses_val.append(loss_val)
    
plt.plot(losses_val)
plt.title('Loss on the validation set')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()

print(sess.run([w]))
```


    <matplotlib.figure.Figure at 0x7f269c7263c8>


    [array([[4.9959269e+00],
           [9.4965822e-04],
           [3.2372758e+00]], dtype=float32)]


By running this piece of code you should see a result close to this:
```
[4.9924135, 0.00040895029, 3.4504161]
```
Which is a relatively close approximation to our parameters.

This is just tip of the iceberg for what TensorFlow can do. Many problems such as optimizing large neural networks with millions of parameters can be implemented efficiently in TensorFlow in just a few lines of code. TensorFlow takes care of scaling across multiple devices, and threads, and supports a variety of platforms.

## Understanding static and dynamic shapes
<a name="shapes"></a>
Tensors in TensorFlow have a static shape attribute which is determined during graph construction. The static shape may be underspecified. For example we might define a tensor of shape [None, 128]:


```python
import tensorflow as tf


a = tf.placeholder(tf.float32, [None, 128])
```

This means that the first dimension can be of any size and will be determined dynamically during Session.run(). You can query the static shape of a Tensor as follows:


```python
a.shape.as_list()
```




    [None, 128]



To get the dynamic shape of the tensor you can call tf.shape op, which returns a tensor representing the shape of the given tensor:


```python
tf.shape(a)
```




    <tf.Tensor 'Shape:0' shape=(2,) dtype=int32>



The static shape of a tensor can be set with Tensor.set_shape() method:


```python
a.set_shape([32, 128])  # static shape of a is [32, 128]
a.set_shape([None, 128])  # first dimension of a is determined dynamically
```

You can reshape a given tensor dynamically using tf.reshape function:


```python
a =  tf.reshape(a, [32, 128])
a
```




    <tf.Tensor 'Reshape:0' shape=(32, 128) dtype=float32>



It can be convenient to have a function that returns the static shape when available and dynamic shape when it's not. The following utility function does just that:


```python
def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims
```

Now imagine we want to convert a Tensor of rank 3 to a tensor of rank 2 by collapsing the second and third dimensions into one. We can use our get_shape() function to do that:


```python
b = tf.placeholder(tf.float32, [None, 10, 32])
shape = get_shape(b)
b = tf.reshape(b, [shape[0], shape[1] * shape[2]])
b
```




    <tf.Tensor 'Reshape_1:0' shape=(?, 320) dtype=float32>



Note that this works whether the shapes are statically specified or not.

In fact we can write a general purpose reshape function to collapse any list of dimensions:


```python
import tensorflow as tf
import numpy as np


def reshape(tensor, dims_list):
    shape = get_shape(tensor)
    dims_prod = []
    for dims in dims_list:
        if isinstance(dims, int):
            dims_prod.append(shape[dims])
        elif all([isinstance(shape[d], int) for d in dims]):
            dims_prod.append(np.prod([shape[d] for d in dims]))
        else:
            dims_prod.append(tf.prod([shape[d] for d in dims]))
    tensor = tf.reshape(tensor, dims_prod)
    return tensor
```

Then collapsing the second dimension becomes very easy:


```python
b = tf.placeholder(tf.float32, [None, 10, 32])
b = reshape(b, [0, [1, 2]])
b
```




    <tf.Tensor 'Reshape_2:0' shape=(?, 320) dtype=float32>



## Scopes and when to use them
<a name="scopes"></a>

Variables and tensors in TensorFlow have a name attribute that is used to identify them in the symbolic graph. If you don't specify a name when creating a variable or a tensor, TensorFlow automatically assigns a name for you:


```python
a = tf.constant(1)
print(a.name)

b = tf.Variable(1)
print(b.name)
```

    Const:0
    Variable:0


You can overwrite the default name by explicitly specifying it:


```python
a = tf.constant(1, name="a")
print(a.name)

b = tf.Variable(1, name="b")
print(b.name)
```

    a:0
    b:0


TensorFlow introduces two different context managers to alter the name of tensors and variables. The first is tf.name_scope:


```python
with tf.name_scope("scope"):
    a = tf.constant(1, name="a")
    print(a.name)

    b = tf.Variable(1, name="b")
    print(b.name)

    c = tf.get_variable(name="c", shape=[])
    print(c.name)
```

    scope/a:0
    scope/b:0
    c:0


Note that there are two ways to define new variables in TensorFlow, by creating a tf.Variable object or by calling tf.get_variable. Calling tf.get_variable with a new name results in creating a new variable, but if a variable with the same name exists it will raise a ValueError exception, telling us that re-declaring a variable is not allowed.

tf.name_scope affects the name of tensors and variables created with tf.Variable, but doesn't impact the variables created with tf.get_variable.

Unlike tf.name_scope, tf.variable_scope modifies the name of variables created with tf.get_variable as well:


```python
with tf.variable_scope("scope"):
    a = tf.constant(1, name="a")
    print(a.name)

    b = tf.Variable(1, name="b")
    print(b.name)

    c = tf.get_variable(name="c", shape=[])
    print(c.name)
```

    scope_1/a:0
    scope_1/b:0
    scope/c:0



```python
with tf.variable_scope("scope"):
    a1 = tf.get_variable(name="a", shape=[])
    # Disallowed
    # a2 = tf.get_variable(name="a", shape=[])
```

But what if we actually want to reuse a previously declared variable? Variable scopes also provide the functionality to do that:


```python
# We've defined a1 above, so in order to create it again, we need to reset
# the default graph
tf.reset_default_graph()

with tf.variable_scope("scope"):
    a1 = tf.get_variable(name="a", shape=[])
    
with tf.variable_scope("scope", reuse=True):
    a2 = tf.get_variable(name="a", shape=[])  # OK
```

This becomes handy for example when using built-in neural network layers:


```python
image1 = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
image2 = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])

conv1_weights = tf.get_variable('conv1_w', [3, 3, 3, 64])
features1 = tf.nn.conv2d(image1, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')

# Use the same convolution weights to process the second image
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    conv1_weights = tf.get_variable('conv1_w')
    features2 = tf.nn.conv2d(image2, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
```

This syntax may not look very clean to some. Especially if you want to do lots of variable sharing keeping track of when to define new variables and when to reuse them can be cumbersome and error prone. TensorFlow templates are designed to handle this automatically:


```python
conv3x32 = tf.make_template("conv3x32", lambda x: tf.layers.conv2d(x, 32, 3))
features1 = conv3x32(image1)
features2 = conv3x32(image2)  # Will reuse the convolution weights.
```

You can turn any function to a TensorFlow template. Upon the first call to a template, the variables defined inside the function would be declared and in the consecutive invocations they would automatically get reused.

## Broadcasting the good and the ugly
<a name="broadcast"></a>
TensorFlow supports broadcasting elementwise operations. Normally when you want to perform operations like addition and multiplication, you need to make sure that shapes of the operands match, e.g. you can’t add a tensor of shape [3, 2] to a tensor of shape [3, 4]. But there’s a special case and that’s when you have a singular dimension. TensorFlow implicitly tiles the tensor across its singular dimensions to match the shape of the other operand. So it’s valid to add a tensor of shape [3, 2] to a tensor of shape [3, 1]


```python
a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[1.], [2.]])
# c = a + tf.tile(b, [1, 2])
c = a + b
```

Broadcasting allows us to perform implicit tiling which makes the code shorter, and more memory efficient, since we don’t need to store the result of the tiling operation. One neat place that this can be used is when combining features of varying length. In order to concatenate features of varying length we commonly tile the input tensors, concatenate the result and apply some nonlinearity. This is a common pattern across a variety of neural network architectures:


```python
a = tf.random_uniform([5, 3, 5])
b = tf.random_uniform([5, 1, 6])

# concat a and b and apply nonlinearity
tiled_b = tf.tile(b, [1, 3, 1])
c = tf.concat([a, tiled_b], 2)
d = tf.layers.dense(c, 10, activation=tf.nn.relu)
```

But this can be done more efficiently with broadcasting. We use the fact that f(m(x + y)) is equal to f(mx + my). So we can do the linear operations separately and use broadcasting to do implicit concatenation:


```python
pa = tf.layers.dense(a, 10, activation=None)
pb = tf.layers.dense(b, 10, activation=None)
d = tf.nn.relu(pa + pb)
```

In fact this piece of code is pretty general and can be applied to tensors of arbitrary shape as long as broadcasting between tensors is possible:


```python
def merge(a, b, units, activation=tf.nn.relu):
    pa = tf.layers.dense(a, units, activation=None)
    pb = tf.layers.dense(b, units, activation=None)
    c = pa + pb
    if activation is not None:
        c = activation(c)
    return c
```

A slightly more general form of this function is [included](#merge) in the cookbook.

So far we discussed the good part of broadcasting. But what’s the ugly part you may ask? Implicit assumptions almost always make debugging harder to do. Consider the following example:


```python
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b)
```

What do you think the value of c would be after evaluation? If you guessed 6, that’s wrong. It’s going to be 12. This is because when rank of two tensors don’t match, TensorFlow automatically expands the first dimension of the tensor with lower rank before the elementwise operation, so the result of addition would be [[2, 3], [3, 4]], and the reducing over all parameters would give us 12.


```python
sess = tf.Session()
sess.run(c)
```




    12.0



The way to avoid this problem is to be as explicit as possible. Had we specified which dimension we would want to reduce across, catching this bug would have been much easier:


```python
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b, 0)
sess.run(c)
```




    array([5., 7.], dtype=float32)



Here the value of c is [5, 7], and we immediately would guess based on the shape of the result that there’s something wrong. A general rule of thumb is to always specify the dimensions in reduction operations and when using tf.squeeze.

## Feeding data to TensorFlow
<a name="data"></a>

TensorFlow is designed to work efficiently with large amount of data. So it's important not to starve your TensorFlow model in order to maximize its performance. There are various ways that you can feed your data to TensorFlow.

### Constants
The simplest approach is to embed the data in your graph as a constant:


```python
actual_data = np.random.normal(size=[100])

data = tf.constant(actual_data)
```

This approach can be very efficient, but it's not very flexible. One problem with this approach is that, in order to use your model with another dataset you have to rewrite the graph. Also, you have to load all of your data at once and keep it in memory which would only work with small datasets.

### Placeholders
Using placeholders solves both of these problems:


```python
data = tf.placeholder(tf.float32)

prediction = tf.square(data) + 1

actual_data = np.random.normal(size=[100])

tf.Session().run(prediction, feed_dict={data: actual_data})
```




    array([1.5884947, 1.3410461, 1.014448 , 1.0088717, 1.684809 , 3.0931547,
           1.5160918, 3.546394 , 1.0058033, 1.0689459, 1.6702695, 1.092588 ,
           1.1949763, 1.7730119, 2.1193926, 1.0000746, 1.0013407, 3.1080537,
           2.137538 , 1.0310628, 1.2136636, 1.2966591, 2.7791133, 1.0752715,
           1.0001731, 1.0154577, 1.5353656, 1.2691123, 5.810177 , 1.2699649,
           1.127784 , 1.4303801, 2.648038 , 2.317832 , 1.1239064, 2.0335813,
           3.1863074, 1.0259023, 3.2779973, 2.7894404, 1.7248772, 1.3147612,
           1.6885958, 1.0205342, 1.6476911, 2.663628 , 1.655623 , 1.0092949,
           2.9211397, 1.0660866, 1.825105 , 3.0801072, 2.2063215, 1.0676146,
           2.0505662, 2.1774657, 2.7993016, 1.3100938, 2.1450522, 2.9252868,
           2.6245646, 4.977445 , 1.010375 , 2.449447 , 1.1088196, 1.0540252,
           1.019463 , 1.6646936, 1.097413 , 1.1366236, 1.2116996, 3.2758334,
           5.2763305, 1.2135888, 1.0138184, 1.1088861, 1.4853244, 1.002463 ,
           1.0109003, 1.7839735, 3.8420875, 1.0406535, 1.0167717, 1.3886988,
           1.3332536, 1.1386113, 1.9157982, 1.0001085, 1.7181236, 2.0487337,
           1.0084076, 2.337519 , 1.5753618, 1.0135951, 3.020419 , 1.025023 ,
           2.1717057, 3.6296546, 3.1894002, 1.0283245], dtype=float32)



Placeholder operator returns a tensor whose value is fetched through the feed_dict argument in Session.run function. Note that running Session.run without feeding the value of data in this case will result in an error.

### Python ops
Another approach to feed the data to TensorFlow is by using Python ops:


```python
def py_input_fn():
    actual_data = np.random.normal(size=[100])
    return actual_data

data = tf.py_func(py_input_fn, [], (tf.float32))
```

Python ops allow you to convert a regular Python function to a TensorFlow operation.

### Dataset API
The recommended way of reading the data in TensorFlow however is through the dataset API:


```python
actual_data = np.random.normal(size=[100])
dataset = tf.data.Dataset.from_tensor_slices(actual_data)
data = dataset.make_one_shot_iterator().get_next()
```

If you need to read your data from file, it may be more efficient to write it in TFrecord format and use TFRecordDataset to read it:

```python
dataset = tf.contrib.data.TFRecordDataset(path_to_data)
```

See the [official docs](https://www.tensorflow.org/api_guides/python/reading_data#Reading_from_files) for an example of how to write your dataset in TFrecord format.

Dataset API allows you to make efficient data processing pipelines easily. For example this is how we process our data in the accompanied framework (See
[trainer.py](https://github.com/vahidk/TensorflowFramework/blob/master/trainer.py)):

```python
dataset = ...
dataset = dataset.cache()
if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(batch_size * 5)
dataset = dataset.map(parse, num_threads=8)
dataset = dataset.batch(batch_size)
```

After reading the data, we use Dataset.cache method to cache it into memory for improved efficiency. During the training mode, we repeat the dataset indefinitely. This allows us to process the whole dataset many times. We also shuffle the dataset to get batches with different sample distributions. Next, we use the Dataset.map function to perform preprocessing on raw records and convert the data to a usable format for the model. We then create batches of samples by calling Dataset.batch.

## Take advantage of the overloaded operators
<a name="overloaded_ops"></a>
Just like NumPy, TensorFlow overloads a number of python operators to make building graphs easier and the code more readable.

The slicing op is one of the overloaded operators that can make indexing tensors very easy:
```python
z = x[begin:end]  # z = tf.slice(x, [begin], [end-begin])
```

Be very careful when using this op though. The slicing op is very inefficient and often better avoided, especially when the number of slices is high. To understand how inefficient this op can be let's look at an example. We want to manually perform reduction across the rows of a matrix:


```python
import time


x = tf.random_uniform([500, 10])

z = tf.zeros([10])
for i in range(500):
    z += x[i]

sess = tf.Session()

%time sess.run(z);
```

    CPU times: user 1.36 s, sys: 12.1 ms, total: 1.37 s
    Wall time: 1.37 s





    array([257.78635, 259.78177, 255.85266, 265.00082, 242.49118, 253.8807 ,
           249.92174, 250.81635, 243.49988, 246.82202], dtype=float32)



On my MacBook Pro, this took 976 ms to run! The reason is that we are calling the slice op 500 times, which is going to be very slow to run. A better choice would have been to use tf.unstack op to slice the matrix into a list of vectors all at once:


```python
%%time
z = tf.zeros([10])
for x_i in tf.unstack(x):
    z += x_i
```

    CPU times: user 233 ms, sys: 2.95 ms, total: 236 ms
    Wall time: 239 ms


This took 277 ms. Of course, the right way to do this simple reduction is to use tf.reduce_sum op:


```python
%time z = tf.reduce_sum(x, axis=0)
```

    CPU times: user 2.28 ms, sys: 554 µs, total: 2.83 ms
    Wall time: 2.48 ms


This took 3.36 ms, which is ~300x faster than the original implementation.

TensorFlow also overloads a range of arithmetic and logical operators:


```python
x = tf.random_uniform([500, 10])
y = tf.random_uniform([500, 10])

z = -x  # z = tf.negative(x)
z = x + y  # z = tf.add(x, y)
z = x - y  # z = tf.subtract(x, y)
z = x * y  # z = tf.mul(x, y)
z = x / y  # z = tf.div(x, y)
z = x // y  # z = tf.floordiv(x, y)
z = x % y  # z = tf.mod(x, y)
z = x ** y  # z = tf.pow(x, y)
z = x @ tf.transpose(y)  # z = tf.matmul(x, y)
z = x > y  # z = tf.greater(x, y)
z = x >= y  # z = tf.greater_equal(x, y)
z = x < y  # z = tf.less(x, y)
z = x <= y  # z = tf.less_equal(x, y)
z = abs(x)  # z = tf.abs(x)


# Now, for logical operations
x = tf.constant([True, False])
y = tf.constant([True, True])

z = x & y  # z = tf.logical_and(x, y)
z = x | y  # z = tf.logical_or(x, y)
z = x ^ y  # z = tf.logical_xor(x, y)
z = ~x  # z = tf.logical_not(x)
```

You can also use the augmented version of these ops. For example `x += y` and `x **= 2` are also valid.

Note that Python doesn't allow overloading "`and`", "`or`", and "`not`" keywords.

TensorFlow also doesn't allow using tensors as booleans, as it may be error prone:


```python
x = tf.constant(1.)
# This would raise a TypeError error
# if x: 
#    ...
```

You can either use tf.cond(x, ...) if you want to check the value of the tensor, or use "if x is None" to check the value of the variable.

Other operators that aren't supported are equal (`==`) and not equal (`!=`) operators which are overloaded in NumPy but not in TensorFlow. Use the function versions instead which are `tf.equal` and `tf.not_equal`.


## Understanding order of execution and control dependencies
<a name="control_deps"></a>
As we discussed in the first item, TensorFlow doesn't immediately run the operations that are defined but rather creates corresponding nodes in a graph that can be evaluated with Session.run() method. This also enables TensorFlow to do optimizations at run time to determine the optimal order of execution and possible trimming of unused nodes. If you only have tf.Tensors in your graph you don't need to worry about dependencies but you most probably have tf.Variables too, and tf.Variables make things much more difficult. My advice to is to only use Variables if Tensors don't do the job. This might not make a lot of sense to you now, so let's start with an example.


```python
a = tf.constant(1)
b = tf.constant(2)
a = a + b

sess.run(a)
```




    3



Evaluating "a" will return the value 3 as expected.  Note that here we are creating 3 tensors, two constant tensors and another tensor that stores the result of the addition. Note that you can't overwrite the value of a tensor. If you want to modify it you have to create a new tensor. As we did here.

***
__TIP__: If you don't define a new graph, TensorFlow automatically creates a graph for you by default. You can use tf.get_default_graph() to get a handle to the graph. You can then inspect the graph, for example by printing all its tensors:


```python
# The list of tensors defined so far is quite long, so I'm showing only
# the last 10 elements
tf.contrib.graph_editor.get_tensors(tf.get_default_graph())[-10:]
```




    [<tf.Tensor 'or:0' shape=(2,) dtype=bool>,
     <tf.Tensor 'xor/LogicalOr:0' shape=(2,) dtype=bool>,
     <tf.Tensor 'xor/LogicalAnd:0' shape=(2,) dtype=bool>,
     <tf.Tensor 'xor/LogicalNot:0' shape=(2,) dtype=bool>,
     <tf.Tensor 'xor:0' shape=(2,) dtype=bool>,
     <tf.Tensor 'LogicalNot:0' shape=(2,) dtype=bool>,
     <tf.Tensor 'Const_10:0' shape=() dtype=float32>,
     <tf.Tensor 'Const_11:0' shape=() dtype=int32>,
     <tf.Tensor 'Const_12:0' shape=() dtype=int32>,
     <tf.Tensor 'add_1006:0' shape=() dtype=int32>]



Unlike tensors, variables can be updated. So let's see how we may use variables to do the same thing:


```python
a = tf.Variable(1)
b = tf.constant(2)
assign = tf.assign(a, a + b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(assign))
```

    3


Again, we get 3 as expected. Note that tf.assign returns a tensor representing the value of the assignment.
So far everything seemed to be fine, but let's look at a slightly more complicated example:


```python
a = tf.Variable(1)
b = tf.constant(2)
c = a + b

assign = tf.assign(a, 5)

sess = tf.Session()
for i in range(10):
    sess.run(tf.global_variables_initializer())
    print(sess.run([assign, c]))
```

    [5, 3]
    [5, 7]
    [5, 3]
    [5, 7]
    [5, 7]
    [5, 7]
    [5, 3]
    [5, 3]
    [5, 7]
    [5, 7]


Note that the tensor c here won't have a deterministic value. This value might be 3 or 7 depending on whether addition or assignment gets executed first.

You should note that the order that you define ops in your code doesn't matter to TensorFlow runtime. The only thing that matters is the control dependencies. Control dependencies for tensors are straightforward. Every time you use a tensor in an operation that op will define an implicit dependency to that tensor. But things get complicated with variables because they can take many values.

When dealing with variables, you may need to explicitly define dependencies using tf.control_dependencies() as follows:


```python
a = tf.Variable(1)
b = tf.constant(2)
c = a + b

with tf.control_dependencies([c]):
    assign = tf.assign(a, 5)

sess = tf.Session()
for i in range(10):
    sess.run(tf.global_variables_initializer())
    print(sess.run([assign, c]))
```

    [5, 3]
    [5, 3]
    [5, 3]
    [5, 3]
    [5, 3]
    [5, 3]
    [5, 3]
    [5, 3]
    [5, 3]
    [5, 3]

