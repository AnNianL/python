import numpy as np
import tensorflow as tf
import cv2

# 定义输入和输出的大小
input_size = (416, 416, 3)
output_sizes = [(13, 13, 5, 25), (26, 26, 5, 25), (52, 52, 5, 25)]

# 定义类别数和锚框尺寸
num_classes = 80
anchor_sizes = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)]

# 定义 YOLOv2 模型
inputs = tf.keras.Input(shape=input_size)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False)(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

for i in range(5):
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

for i in range(5):
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

# 输出部分
yolo_outputs = []
for i, output_size in enumerate(output_sizes):
    output = tf.keras.layers.Conv2D(filters=output_size[2] * (num_classes + 5), kernel_size=1, strides=1, padding='same', use_bias=True)(x)
    yolo_outputs.append(output)

model = tf.keras.Model(inputs=inputs, outputs=yolo_outputs)

# 定义损失函数
def yolo_loss(output_sizes, num_classes, anchor_sizes):
    def loss(y_true, y_pred):
        # 解码预测结果
        grid_size = input_size[0] // output_sizes[0][0]
        ratio = [1, 2, 4]
        bbox_xywh_pred, class_probs_pred = [], []
        input_shape = tf.shape(y_pred[0])[1:3] * grid_size
        for i, output_size in enumerate(output_sizes):
            grid_size = input_size[0] // output_size[0]
            ratio = anchor_sizes[i]
            anchor_size = anchor_sizes[i] * grid_size
            num_anchors = output_size[2]

            output = tf.reshape(y_pred[i], [-1, output_size[0], output_size[1], num_anchors, num_classes + 5])

            bbox_xy_pred = output[..., 0:2]
            bbox_wh_pred = output[..., 2:4]
            bbox_confidence_pred = output[..., 4:5]
            class_probs_pred = output[..., 5:]

            bbox_xy = tf.keras.backend.sigmoid(bbox_xy_pred)
            bbox_wh = tf.keras.backend.exp(bbox_wh_pred) * anchor_size
            bbox_confidence = tf.keras.backend.sigmoid(bbox_confidence_pred)

            # 计算预测框在输入图像上的坐标和大小
            bbox_xy_grid = tf.meshgrid(tf.range(output_size[0]), tf.range(output_size[1]))
            bbox_xy_grid = tf.expand_dims(tf.stack(bbox_xy_grid, axis=-1), axis=2)
            bbox_xy_grid = tf.cast(bbox_xy_grid, tf.keras.backend.dtype(y_true))

            bbox_xy = bbox_xy + bbox_xy_grid
            bbox_xy = bbox_xy / tf.cast(input_shape[::-1], tf.keras.backend.dtype(y_true))
            bbox_wh = bbox_wh / tf.cast(input_shape[::-1], tf.keras.backend.dtype(y_true))

            bbox_xywh_pred.append(tf.keras.backend.concatenate([bbox_xy, bbox_wh], axis=-1))
            class_probs_pred.append(class_probs_pred)

        # 将预测结果转换为张量
        bbox_xywh_pred = tf.keras.backend.concatenate(bbox_xywh_pred, axis=1)
        class_probs_pred = tf.keras.backend.concatenate(class_probs_pred, axis=1)

        # 计算 ground truth
        bbox_xywh_true = y_true[..., 0:4]
        class_probs_true = y_true[..., 4:]

        object_mask = y_true[..., 4:5]
        ignore_mask = tf.TensorArray(tf.keras.backend.dtype(y_true), size=1, dynamic_size=True)
        detection_masks = tf.TensorArray(tf.keras.backend.dtype(y_true), size=1, dynamic_size=True)
        input_shape = tf.cast(input_shape, tf.keras.backend.dtype(y_true))
        for i in tf.range(tf.shape(y_true)[0]):
            true_box_xywh = tf.boolean_mask(bbox_xywh_true[i, ..., 0:4], tf.cast(object_mask[i, ..., 0], 'bool'))
            iou = box_iou(bbox_xywh_pred[i], true_box_xywh)
            best_iou = tf.reduce_max(iou, axis=-1)
            ignore_mask_box = tf.cast(best_iou < 0.5, tf.keras.backend.dtype(y_true))
            ignore_mask = ignore_mask.write(i, ignore_mask_box)

            for j in tf.range(tf.shape(true_box_xywh)[0]):
                detection_mask = tf.TensorArray(tf.keras.backend.dtype(y_true), size=1, dynamic_size=True)
                box_xywh = bbox_xywh_true[i, j:j+1]
                iou = box_iou(bbox_xywh_pred[i], box_xywh)
                best_anchor = tf.cast(tf.argmax(iou, axis=-1), tf.keras.backend.dtype(y_true))
                detection_mask = detection_mask.write(best_anchor, tf.cast(1.0, tf.keras.backend.dtype(y_true)))
                detection_masks = detection_masks.write(i, detection_mask.stack())

        ignore_mask = ignore_mask.stack()
        detection_masks = detection_masks.stack()
        bbox_confidence_true = detection_masks
        class_probs_true = tf.keras.backend.one_hot(tf.cast(class_probs_true, tf.keras.backend.dtype(y_pred)), num_classes=num_classes)

        # 计算损失函数
        xy_loss = object_mask * tf.square(bbox_xywh_true[..., 0:2] - bbox_xywh_pred[..., 0:2])
        wh_loss = object_mask * tf.square(bbox_xywh_true[..., 2:4] - bbox_xywh_pred[..., 2:4])
        confidence_loss = object_mask * tf.square(1 - bbox_confidence_pred) + \
                          ignore_mask * tf.square(0 - bbox_confidence_pred)
        class_probs_loss = object_mask * tf.square(class_probs_true - class_probs_pred)

        xy_loss = tf.reduce_sum(xy_loss) / tf.cast(tf.shape(y_true)[0], tf.keras.backend.dtype(y_pred))
        wh_loss = tf.reduce_sum(wh_loss) / tf.cast(tf.shape(y_true)[0], tf.keras.backend.dtype(y_pred))
        confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(tf.shape(y_true)[0], tf.keras.backend.dtype(y_pred))
        class_probs_loss = tf.reduce_sum(class_probs_loss) / tf.cast(tf.shape(y_true)[0], tf.keras.backend.dtype(y_pred))

        return xy_loss + wh_loss + confidence_loss + class_probs_loss

    return loss

# 定义 box_iou 函数，用于计算预测框和真实框之间的 iou
def box_iou(b1, b2):
    bx1, by1, bw1, bh1 = tf.split(b1, 4, axis=-1)
    bx2, by2, bw2, bh2 = tf.split(b2, 4, axis=-1)

    area1 = bw1 * bh1
    area2 = bw2 * bh2

    xmin = tf.keras.backend.maximum(bx1 - bw1 / 2.0, bx2 - bw2 / 2.0)
    ymin = tf.keras.backend.maximum(by1 - bh1 / 2.0, by2 - bh2 / 2.0)
    xmax = tf.keras.backend.minimum(bx1 + bw1 / 2.0, bx2 + bw2 / 2.0)
    ymax = tf.keras.backend.minimum(by1 + bh1 / 2.0, by2 + bh2 / 2.0)

    inter_w = tf.keras.backend.maximum(0.0, xmax - xmin)
    inter_h = tf.keras.backend.maximum(0.0, ymax - ymin)

    inter_area = inter_w * inter_h
    union_area = area1 + area2 - inter_area

    return inter_area / union_area

# 加载模型和权重
model = tf.keras.models.load_model('path/to/weights.h5', compile=False)
model.compile(optimizer='adam', loss=yolo_loss(output_sizes, num_classes, anchor_sizes))

# 加载图像文件
img = cv2.imread('path/to/image.jpg')
img = cv2.resize(img, (input_size[0], input_size[1]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0

# 对图像进行预处理
img_input = np.expand_dims(img, axis=0)

# 预测物体位置
predictions = model.predict(img_input)

# 输出检测结果
for i, prediction in enumerate(predictions):
    grid_size = input_size[0] // output_sizes[i][0]
    bbox_xywh_pred = prediction[..., 0:4]
    class_probs_pred = prediction[..., 5:]

    bbox_xywh_pred[..., 0] = bbox_xywh_pred[..., 0] * grid_size
    bbox_xywh_pred[..., 1] = bbox_xywh_pred[..., 1] * grid_size
    bbox_xywh_pred[..., 2] = bbox_xywh_pred[..., 2] * anchor_sizes[i][0]
    bbox_xywh_pred[..., 3] = bbox_xywh_pred[..., 3] * anchor_sizes[i][1]

    # 对每个网格单元以及锚框进行非极大抑制
    bbox_confidences = tf.keras.backend.sigmoid(prediction[..., 4:5])
    class_probs = tf.keras.backend.softmax(class_probs_pred, axis=-1)
    scores = bbox_confidences * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox_xywh_pred, (tf.shape(bbox_xywh_pred)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.5,
        score_threshold=0.5
        )
    # 输出检测结果
for i in range(valid_detections[0]):
    print('Detected object {0}: class "{1}", confidence: {2}, '
          'coordinates: {3}'.format(i + 1, classes[0][i], scores[0][i], boxes[0][i].numpy()))




