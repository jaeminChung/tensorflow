import tensorflow as tf

def linear_regression1():
    x_data = [1., 2., 3.]
    y_data = [1., 2., 3.]
    
    # try to find values for w and b that compute y_data = W * x_data + b
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    
    # my hypothesis
    hypothesis = W * x_data + b
    
    # Simplified cost function
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    
    # minimize
    rate = tf.Variable(0.1)  # learning rate, alpha
    optimizer = tf.train.GradientDescentOptimizer(rate)
    train = optimizer.minimize(cost)
    
    # before starting, initialize the variables. We will 'run' this first.
    init = tf.initialize_all_variables()
    
    # launch the graph
    sess = tf.Session()
    sess.run(init)
    
    # fit the line
    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print('{:4} {} {} {}'.format(step, sess.run(cost), sess.run(W), sess.run(b)))
    
    # learns best fit is W: [1] b: [0]


def linear_regression_with_placeholder():
    x_data = [1., 2., 3., 4.]
    y_data = [2., 4., 6., 8.]
    
    # range is -100 ~ 100
    W = tf.Variable(tf.random_uniform([1], -100., 100.))
    b = tf.Variable(tf.random_uniform([1], -100., 100.))
    
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    
    hypothesis = W * X + b
    
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    rate = tf.Variable(0.1)
    optimizer = tf.train.GradientDescentOptimizer(rate)
    train = optimizer.minimize(cost)
    
    init = tf.initialize_all_variables()
    
    sess = tf.Session()
    sess.run(init)
    
    for step in range(2001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))
    
    print(sess.run(hypothesis, feed_dict={X: 5}))           # [ 10.]
    print(sess.run(hypothesis, feed_dict={X: 2.5}))         # [5.]
    print(sess.run(hypothesis, feed_dict={X: [2.5, 5]}))    # [  5.  10.], 원하는 X의 값만큼 전달.


linear_regression1()
linear_regression_with_placeholder()
