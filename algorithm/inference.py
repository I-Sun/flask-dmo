# encoding: utf-8
import argparse
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import visualization_utils as vis_util, label_map_util
import base64
import re
from cStringIO import StringIO

# if tf.__version__ < '1.4.0':
#    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


NUM_CLASSES = 3
CKPT_PATH = "/home/insun/models-r1.5/research/data/udacity/model/frozen_inference_graph.pb"
LABEL_PATH = "/home/insun/models-r1.5/research/data/udacity/model/labels_items.txt"

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


def _load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def _base64_to_image(base64_str, image_path=None):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    binary_data = base64.b64decode(base64_data)
    img_data = StringIO(binary_data)
    img = Image.open(img_data)
    if image_path:
        img.save(image_path)
    return img


def image_to_base64(image_path):
    img = Image.open(image_path)
    output_buffer = StringIO()
    img.save(output_buffer, format='JPEG')
    binary_data = output_buffer.getvalue()
    base64_data = base64.b64encode(binary_data)
    return base64_data


def _array_to_base64(image_np):
    img = Image.fromarray(image_np.astype('uint8')).convert('RGB')
    output_buffer = StringIO()
    img.save(output_buffer, format='JPEG')
    binary_data = output_buffer.getvalue()
    base64_data = base64.b64encode(binary_data)
    return base64_data


def detect_car(image, output_path=None):
    image = _base64_to_image(image)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(CKPT_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    label_map = label_map_util.load_labelmap(LABEL_PATH)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image_np = _load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            _, t_boxs = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            if output_path:
                plt.imsave(output_path, image_np)

            print("-----------------------------------------")
            print("boxes : ", t_boxs)

            return "data:image/jpeg;base64,%s"%_array_to_base64(image_np)



def infernece_img(ckpt_path,                # 绝对路径 /exported_graphs/frozen_inference_graph.pb
            label_path,                 # 绝对路径 /output/labels_items.txt
            infernece_file,             # 输入文件路径
            output_file,                # 输出文件路径
            class_num):                 # 输出类型数量                

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=class_num, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image = Image.open(infernece_file)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            _, t_boxs = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            plt.imsave(output_file, image_np)
            print("-----------------------------------------")
            print("boxes : ", t_boxs)
            
            
def main(_):
    FLAGS, unparsed = parse_args()

    PATH_TO_CKPT = os.path.join(FLAGS.output_dir, 'exported_graphs/frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'labels_items.txt')
    test_img_path = os.path.join(FLAGS.dataset_dir, '1479498581479711710.jpg')
    output_file = os.path.join(FLAGS.output_dir, 'output.png')
    infernece_img(PATH_TO_CKPT,                     
            PATH_TO_LABELS,                     
            test_img_path,
            output_file,
            NUM_CLASSES)        


if __name__=='__main__':
    tf.app.run()