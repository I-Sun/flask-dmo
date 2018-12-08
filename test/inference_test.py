from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import inference

flags = tf.app.flags
flags.DEFINE_string('ckpt_path','','model path')
flags.DEFINE_string('label_path', '', '')
flags.DEFINE_string('inference_folder', '', '')
flags.DEFINE_string('output_folder', '~/tmp/output', '')
flags.DEFINE_integer('class_num', 3, '')

FLAGS = flags.FLAGS

def main(_):
    ckpt_path = FLAGS.ckpt_path
    label_path = FLAGS.label_path
    inference_folder = FLAGS.inference_folder
    output_folder = FLAGS.output_folder
    class_num = FLAGS.class_num
    counter = 0
    filelist = os.listdir(inference_folder)
    print("total : ", len(filelist))
    for file in filelist:      
        abosult_path = os.path.join(inference_folder, file)
        if file.endswith(".jpg"):
            prefix = file[:-4]
            output_filename = prefix + "_out.jpg"
            output_filename = os.path.join(output_folder, output_filename)
            counter = counter + 1
            inference.infernece_img(ckpt_path, 
                        label_path,
                        abosult_path,
                        output_filename,
                        class_num)
            print("handle file : ", counter, " ", file)

if __name__=='__main__':
    tf.app.run()
# python insun_code/inference_test.py --ckpt_path=/home/insun/models-r1.5/research/data/udacity/model/frozen_inference_graph.pb --label_path=/home/insun/models-r1.5/research/data/udacity/model/labels_items.txt --inference_folder=/home/insun/models-r1.5/research/data/udacity/inference-in  --output_folder=/home/insun/models-r1.5/research/data/udacity/inference-out
