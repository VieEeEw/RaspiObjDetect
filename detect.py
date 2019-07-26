class PythonVersionError(Exception):
    pass


import sys

if float(sys.version[:3]) < 3.6:
    raise PythonVersionError("Need Python version>=3.6, get" + sys.version[:3])

from utils import visualization_utils as vis_util
from utils import label_map_util
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import numpy as np

__VERSION__ = 1.3


def parse_arg():
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--data-path', default=0, help='data path to image or video, default to 0(camera)')
    args.add_argument('-pb', '--pb-path', help='frozen inference graph file')
    args.add_argument('-lp', '--label-path', default='label_map.pbtxt', help='labels dictionary path')
    args.add_argument('-v', '--video', action='store_true', default=False, help='using video or not')
    args.add_argument('-p', '--plot', action='store_true', default=True, help='plot your image or not')
    flag = args.parse_args()
    return flag


def _detect(img, tensor_dict, sess, input_tensor):
    output_dict = sess.run(tensor_dict, feed_dict={input_tensor: img})
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict


def show_detected_img(sess, cls, tensor_dict, input_tensor,
                      img_path=None, img_data=None, min_threshold=0.3, video=False, plot=True):
    if img_path is None and img_data is None:
        raise RuntimeError("At least one img data should be specified")
    img = img_data if img_data is not None else cv2.imread(img_path)
    if img_path is not None:
        if not os.path.exists(img_path):
            raise FileNotFoundError("No image is found at " + img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array([img])
    output_dict = _detect(img, tensor_dict, sess, input_tensor)
    img = img[0]
    _, statuses = vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        cls,
        min_score_thresh=min_threshold,
        use_normalized_coordinates=True,
        line_thickness=3)
    if plot and not video:
        plt.figure(figsize=(24, 16))
        plt.imshow(img)
        plt.show()
    return statuses, img


def main(parsed_args=None, data_path=None, pb_path=None, label_path='label_map.pbtxt', video=False, plot=True):
    print(__VERSION__)
    if parsed_args:
        data_path = parsed_args.data_path or data_path
        pb_path = parsed_args.pb_path or pb_path
        label_path = parsed_args.label_path or label_path
        video = parsed_args.video if parsed_args.video is not None else video
        plot = parsed_args.plot if parsed_args.plot is not None else plot
    if not video and data_path.split('.')[-1] == 'mp4':
        raise RuntimeError("Invalid data type with name " + data_path)
    if data_path != '0' and not os.path.exists(data_path):
        raise FileExistsError(f"Image path not found {data_path}")
    if not os.path.exists(pb_path):
        raise FileExistsError(f"Pb file path not found {pb_path}")
    classes = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)
    with tf.io.gfile.GFile(pb_path, 'rb') as fid:
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
    ops = graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            print(tensor_name)
            tensor_dict[key] = graph.get_tensor_by_name(tensor_name)
    input_tensor = graph.get_tensor_by_name('image_tensor:0')
    with tf.device('gpu:0'):
        with tf.Session(graph=graph) as sess:
            if video:
                vid_cap = cv2.VideoCapture(data_path)
                while vid_cap.isOpened():
                    reval, frame = vid_cap.read()
                    if reval:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        raise RuntimeError("No image!")
                    status, img = show_detected_img(sess, classes, tensor_dict, input_tensor, img_data=frame, video=True)
                    print(status)
                    cv2.imshow('Camera', frame)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
                vid_cap.release()
            else:
                status, _ = show_detected_img(sess, classes, tensor_dict, input_tensor, img_path=data_path, video=False,
                                              plot=plot)
                print(status)


if __name__ == '__main__':
    main(parse_arg())
