import tensorflow as tf

# Declare value, datatype and shape
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)
rank1_tensor = tf.Variable(["test", "ok", "tim"], tf.string)
rank2_tensor = tf.Variable([["test", "ok"],["test", "yes"]], tf.string)

rank = tf.rank(rank2_tensor)
shape = rank2_tensor.shape
print(rank, shape)

#Reshape
tensor1 = tf.ones([1,2,3])
tensor2 = tf.reshape(tensor1, [2,3,1])
tensor3 = tf.reshape(tensor2, [3, -1]) # it reshapes [3,2] because it needs to be the same shape
tensor4 = tf.zeros([5,5,5,5])
tensor5 = tf.reshape(tensor4, [125, -1])
print(tensor5)

#Evaluate tensors

with tf.Session() as sess:
    tf.eval()