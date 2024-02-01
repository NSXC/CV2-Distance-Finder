import cv2
import numpy as np
import tensorflow as tf
model = tf.saved_model.load('SSD Mobile Net Model Path')#ADD BEFORE RUNNING
camera_matrix_1 = np.load("camera_matrix_1.npy")
dist_coeff_1 = np.load("dist_coeff_1.npy")
fov_x_1 = np.deg2rad(62.2)  
camera_matrix_2 = np.load("camera_matrix_2.npy")
dist_coeff_2 = np.load("dist_coeff_2.npy")
fov_x_2 = np.deg2rad(62.2)  
known_object_width_inches = 12.0  #AJUST AS NEEDED USED IN QUCIK CALIBRATE
cap_1 = cv2.VideoCapture(0)  
cap_2 = cv2.VideoCapture(1) 

while True:
    ret_1, frame_1 = cap_1.read()
    ret_2, frame_2 = cap_2.read()
    image_np_1 = np.array(frame_1)
    image_np_2 = np.array(frame_2)
    input_tensor_1 = tf.convert_to_tensor(np.expand_dims(image_np_1, 0), dtype=tf.uint8)
    input_tensor_2 = tf.convert_to_tensor(np.expand_dims(image_np_2, 0), dtype=tf.uint8)
    detections_1 = model(input_tensor_1)
    detections_2 = model(input_tensor_2)
    valid_boxes_1 = []
    confidence_scores_1 = []
    for i in range(int(detections_1['num_detections'])):
        box_1 = detections_1['detection_boxes'][0, i].numpy()
        confidence_1 = detections_1['detection_scores'][0, i].numpy()
        if confidence_1 > 0.36:
            valid_boxes_1.append(box_1)
            confidence_scores_1.append(confidence_1)
    valid_boxes_2 = []
    confidence_scores_2 = []
    for i in range(int(detections_2['num_detections'])):
        box_2 = detections_2['detection_boxes'][0, i].numpy()
        confidence_2 = detections_2['detection_scores'][0, i].numpy()
        if confidence_2 > 0.36: #AJUST AS NEEDED
            valid_boxes_2.append(box_2)
            confidence_scores_2.append(confidence_2)

    if valid_boxes_1 and valid_boxes_2:
        max_confidence_index_1 = np.argmax(confidence_scores_1)
        box_1 = valid_boxes_1[max_confidence_index_1]
        max_confidence_index_2 = np.argmax(confidence_scores_2)
        box_2 = valid_boxes_2[max_confidence_index_2]

        object_width_pixels_1 = int((box_1[3] - box_1[1]) * 800)
        distance_1 = (known_object_width_inches * 800) / (2 * object_width_pixels_1 * np.tan(fov_x_1))
        cv2.putText(frame_1, f"Distance: {distance_1:.2f} units", (int(box_1[1] * 800), int(box_1[0] * 600) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        object_width_pixels_2 = int((box_2[3] - box_2[1]) * 800)
        distance_2 = (known_object_width_inches * 800) / (2 * object_width_pixels_2 * np.tan(fov_x_2))
        cv2.putText(frame_2, f"Distance: {distance_2:.2f} units", (int(box_2[1] * 800), int(box_2[0] * 600) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('object detection - Camera 1', cv2.resize(frame_1, (800, 600)))
    cv2.imshow('object detection - Camera 2', cv2.resize(frame_2, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_1.release()
cap_2.release()
cv2.destroyAllWindows()
