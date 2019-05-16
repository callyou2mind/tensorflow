import tensorflow as tf
import numpy as np

W=tf.Variable(initial_value=tf.random_normal([1]), name='weight',trainable=True)
b=tf.Variable(initial_value=0.001,name='bias',trainable=True)

x=tf.placeholder(dtype=tf.float32, shape=[1],name='x')
y=tf.add(tf.multiply(W,x),b,name='output')
init=tf.global_variables_initializer()
saver=tf.train.Saver()
save_path="/home/"
model_save=save_path+"saved_model.ckpt"
#TensorFlow session
with tf.Session() as sess:
    sess.run(init) #initialising the variables
    op=sess.run(y, feed_dict={x: np.reshape(1.5,[1])}) #sample run(optional)
    saver.save(sess,model_save) #saving the model
    tf.train.write_graph(sess.graph_def, save_path, 'saved_model.pbtxt')



import tensorflow as tf
import numpy as np
from tensorflow.python.tools import freeze_graph

save_path="/home/"
MODEL_NAME = 'Sample_model' #name of the model optional
input_graph_path = save_path+'saved_model.pbtxt'#complete path to the input graph
checkpoint_path = save_path+'saved_model.ckpt' #complete path to the model's checkpoint file
input_saver_def_path = ""
input_binary = False
output_node_names = "output" #output node's name. Should match to that mentioned in your code
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = save_path+'saved_model'+'.pb' # the name of .pb file you would like to give
clear_devices = True

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")    

saved_model_dir = '/home'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
