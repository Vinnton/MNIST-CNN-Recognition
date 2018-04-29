import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import struct
import datetime
import matplotlib.pyplot as plt

def load_mnist(train_labels_path,train_images_path,test_labels_path,test_images_path):
    with open(train_labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        train_labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(train_images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        train_images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(train_labels), 784)
    with open(test_labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        test_labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(test_images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        test_images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(test_labels), 784)
    return  train_images, train_labels,test_images,test_labels


train_images, train_labels,test_images,test_labels=load_mnist('train-labels','train-images','test-labels','test-images')
#Normalization
train_images=train_images.astype(np.float)
train_images=(train_images-127)/255
test_images=test_images.astype(np.float)
test_images=(test_images-127)/255

'''
#Softmax Realization
def training(images,labels):
    labels=tf.one_hot(labels,10)#将label变成onehot向量
    TrainImage = tf.placeholder(tf.float32, shape=[None, 784])
    TrainLabel = tf.placeholder(tf.float32, shape=[None, 10])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    hypothsis = tf.nn.softmax(tf.matmul(TrainImage,W) + b)#输出的每一行表示一个image,第i行第j列表示第i个iamge代表数字j的概率（概率及对数字j的预测值除以所有预测值的和）
    hypothsis =hypothsis +1e-10
    #hypothsis =tf.clip_by_value(hypothsis,1e-8,tf.reduce_max(hypothsis))#避免预测值有0的出现，这样log就会出现NaN
    loss_cross_entropy = -tf.reduce_sum(TrainLabel*tf.log(hypothsis))
    Optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-6).minimize(loss_cross_entropy) 
    sess=tf.Session()
    sess.run(tf.initialize_all_variables())
    StartTime=datetime.datetime.now()
    for step in range(400000): 
        try:
            BatchImage=images[(step*50)%60000:(step*50+50)%60000]
            BatchLabel=labels.eval(session=sess)[(step*50)%60000:(step*50+50)%60000]
            Weight,bias,loss,_=sess.run([W,b,loss_cross_entropy,Optimizer], feed_dict={TrainImage:BatchImage, TrainLabel:BatchLabel}) 
            if step%2000==0:
                print("loss",loss)
        except KeyboardInterrupt:
            print("stop")
            break
    FinishTime=datetime.datetime.now()
    print("Time Cost:",(FinishTime-StartTime).total_seconds()/60, "minutes")
    return Weight,bias

def TestAccurancy(Weight,bias,test_labels,test_images):
  hypothsis=np.dot(test_images,Weight)+bias
  prediction = np.equal(np.argmax(hypothsis, 1), test_labels)
  accuracy = np.mean(prediction)
  return accuracy

Weight,bias=training(train_images,train_labels)
Accuracy=TestAccurancy(Weight,bias,test_labels,test_images)
print("Accuracy: ",Accuracy)
'''

#CNN Realization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)#tf.truncated_normal(shape, mean, stddev) 正态分布:shape表示生成张量的维度，mean是均值，stddev是标准差。
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def TrainConv(train_images,train_labels,test_images,test_labels):
    train_images=train_images.reshape(-1,28,28,1)
    train_labels=tf.one_hot(train_labels,10)#将label变成onehot向量
    test_images=test_images.reshape(-1,28,28,1)
    test_labels=tf.one_hot(test_labels,10)#将label变成onehot向量
    TrainImage = tf.placeholder(tf.float32, shape=[None, 28,28,1])
    TrainLabel = tf.placeholder(tf.float32, shape=[None, 10]) 
    #convolutional layer
    #第一层卷积
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    #第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    #因为第二层有64个卷积核，所以对应64个bias
    h_conv1 = tf.nn.relu(conv2d(TrainImage, W_conv1) + b_conv1)#(1, 28, 28, 32) #b_conv1为32，每个卷积核对应一个bias
    h_pool1 = max_pool_2x2(h_conv1)#(1, 14, 14, 32)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)#(1, 7, 7, 64)    
    #Dense connect layer
    W_FullConect1=weight_variable([7 * 7 * 64, 1024])
    b_FullConect1=bias_variable([1024])
    W_FullConect2=weight_variable([1024, 10])
    b_FullConect2=bias_variable([10])
    h_pool2= tf.reshape(h_pool2,[-1,7 * 7 * 64])
    FullConnectLayer1=tf.nn.relu(tf.matmul(h_pool2, W_FullConect1)+b_FullConect1)
    keep_prob = tf.placeholder("float")    #drop out
    FullConnectLayer1Drop = tf.nn.dropout(FullConnectLayer1, keep_prob)#drop out
    hypothesis=tf.nn.softmax(tf.matmul(FullConnectLayer1Drop, W_FullConect2) + b_FullConect2)
    cross_entropy = -tf.reduce_sum(TrainLabel *tf.log(hypothesis))
    Optimizer = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    #预测正确率
    correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(TrainLabel,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    StartTime=datetime.datetime.now()
    ce = []
    for step in range(1000): 
        try:
            BatchImage=train_images[(step*50)%60000:(step*50+50)%60000]
            BatchLabel=train_labels.eval(session=sess)[(step*50)%60000:(step*50+50)%60000]
            A,C,_=sess.run([accuracy,cross_entropy,Optimizer], feed_dict={TrainImage:BatchImage, TrainLabel:BatchLabel, keep_prob:0.9}) 
            ce.append(C)
            if step%1000==0:
                print("accuracy",A)
                print("loss",C)
        except KeyboardInterrupt:
            print("stop")
            break
    FinishTime=datetime.datetime.now()
    print("Time Cost: ",(FinishTime-StartTime).total_seconds()/60,"minutes")
    print("Accuracy: ",sess.run([accuracy], feed_dict={TrainImage:test_images, TrainLabel:test_labels.eval(session=sess), keep_prob:0.9}) )
    return ce            

y = TrainConv(train_images, train_labels,test_images,test_labels)
x = range(1000)
plt.figure()
plt.plot(x,y)
plt.show()


