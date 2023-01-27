import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
tf.set_random_seed(0)

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>6 stride 1        
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x6=>12 stride 2        
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             
#     @ @ @ @ @ @       -- conv. layer 4x4x12=>24 stride 2       
#     ∶∶∶∶∶∶∶∶∶∶∶                                               
#      \x/x\x\x/        -- fully connected layer (relu)         
#       · · · ·                                                 
#       \x/x\x/         -- fully connected layer (softmax)      
#        · · ·                                                  

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# Define paramaters for the model(learning rate, batch size, epochs)
batch_size = 150
training_epochs = 10
learning_rate = tf.Variable (0.001, name = "learning_rate" )

# Create placeholders for features adn labels
# Use None for shape so we can change the batch_size once we've built the graph
# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X')

# Placeholder for correct answers 
Y_ = tf.placeholder(tf.float32, [None, 10], name='Y_')

# three convolutional layers with their channel counts, and a
# fully connected layer (the last layer has 10 softmax neurons)
K = 10  # first convolutional layer output depth

# Create variables for weights and biases (W1,b1,W2,b2)
# Hint: Initialzie weights with small random values between -0.2 and +0.2
# Hint: When using RELUs, make sure biases are initialised with small *positive* values for example 0.1
W1 = tf.Variable(tf.random_uniform([5, 5, 1, K], -0.2, +0.2, tf.float32), name='W1')
b1 = tf.Variable(tf.constant(0.1, tf.float32, [K]), name='b1')

W2 = tf.Variable(tf.zeros([24*24*K, 10]), name='W2')
b2 = tf.Variable(tf.zeros([10]), name='b2')


# Define the model
# Hint: reshape the output from the third convolution for the fully connected layer
# Last layer is defined : loss definition uses it
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, [1, 1, 1, 1], padding='VALID') + b1, name='Y1')
YY = tf.reshape(Y1, shape=[-1, 24*24*K])
Ylogits = tf.matmul(YY, W2) + b2
Y = tf.nn.softmax(Ylogits)

# Define loss and optimizer
# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
loss = tf.reduce_mean(loss)*100

# Define the training step: optimizer
# Use AdamOptimizer instead of GradientDescent
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Evaluate model
# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Visualize TensorBoard
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
merged_summary_op = tf.summary.merge_all()
logs_path='./logs'
summary_writer = tf.summary.FileWriter(logs_path,
                                            graph=tf.get_default_graph())

# Start and do the training 
sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)

    new_lr = learning_rate.assign(learning_rate - 0.00005)
    sess.run(new_lr)

    for i in range(total_batch):
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = mnist.train.next_batch(batch_size)

        # Write data to summary along with _ and c.
        _, c, summary = sess.run([train_step, loss, merged_summary_op], feed_dict={X: batch_X, Y_: batch_Y})

        # Add data to logs.
        summary_writer.add_summary(summary, epoch * total_batch + i)

        avg_cost += c / total_batch
    # Display log per epoch step
    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

print("Test accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))

# Open weight and bias files for reading.
f_handle_W1 = open('W1.BIN','w')
# TODO -> Zad.1 Add the other weights and biases as well. b1, W2 and b2 should be added.
f_handle_b1 = open('b1.BIN','w')
f_handle_W2 = open('W2.BIN','w')
f_handle_b2 = open('b2.BIN','w')

# Evaluate weight and bias values.
W1np = W1.eval(session=sess)
# TODO -> Zad.1 Add the other weights and biases as well. b1, W2 and b2 should be added.
b1np = b1.eval(session=sess)
W2np = W2.eval(session=sess)
b2np = b2.eval(session=sess)

# Transpose the weights.
W1np = np.transpose(W1np)
W2np = np.transpose(W2np)

# Dump weight and bias values to file.
W1np.tofile(f_handle_W1,sep="")
# TODO -> Zad.1 Add the other weights and biases as well. b1, W2 and b2 should be added.
b1np.tofile(f_handle_b1,sep="")
W2np.tofile(f_handle_W2,sep="")
b2np.tofile(f_handle_b2,sep="")


# Close writer
summary_writer.close()

sess.close()

