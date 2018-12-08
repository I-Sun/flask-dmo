
import tensorflow as tf
from algorithm import inference


def main(_):
    inference.infernece_img("/home/insun/models-r1.5/research/data/udacity/model/frozen_inference_graph.pb",
                            "/home/insun/models-r1.5/research/data/udacity/model/labels_items.txt",
                            "/home/insun/models-r1.5/research/data/udacity/inference-in/0000.jpg",
                            "/home/insun/models-r1.5/research/data/udacity/inference-out/0000.jpg",
                            3)


if __name__=='__main__':
    tf.app.run()