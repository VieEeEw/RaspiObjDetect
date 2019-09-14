# import cv2
#
# vid = cv2.VideoCapture(0)
# while vid.isOpened():
#     retval, frame = vid.read()
#     if not retval:
#         break
#     cv2.imshow("result", frame)
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
# vid.release()
from utils import visualization_utils as vis_util
from utils import label_map_util
import tensorflow as tf
import cv2
import numpy as np


def _detect(img, tensor_dict, sess, input_tensor):
    output_dict = sess.run(tensor_dict, feed_dict={input_tensor: img})
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict


def show_detected_img(sess, cls, tensor_dict, input_tensor, img_data, min_threshold=0.3):
    img = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    img = np.array([img])
    output_dict = _detect(img, tensor_dict, sess, input_tensor)
    img = img[0]
    vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        cls,
        min_score_thresh=min_threshold,
        use_normalized_coordinates=True,
        line_thickness=3)
    return img


if __name__ == '__main__':
    version_ = input("Select version from [1, 2]")
    assert version_ in ['1', '2']
    pb_path = 'ssd_mobilenet_v{}.pb'.format(version_)
    label_path = 'label_map.pbtxt'
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
            vid_cap = cv2.VideoCapture(0)
            while vid_cap.isOpened():
                reval, frame = vid_cap.read()
                if reval:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    break
                processed_frame = show_detected_img(sess, classes, tensor_dict, input_tensor, img_data=frame)
                cv2.imshow('Camera', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            vid_cap.release()
