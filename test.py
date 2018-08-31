# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
from config.config_input import PascalVOC2007Input
from config.config_model import DefaultModelConfig
from config.config_train import DefaultTrainConfig

from data_process.data_reader import DataReader
from model.VGGnet_faster_rcnn import VGGNetFasterRCNNModel
from trainval.trainer import Trainer

# define input data
pascal_data_cfg = PascalVOC2007Input()
# define model
vgg_faster_rcnn_cfg = DefaultModelConfig(pascal_data_cfg)
# define train method
default_train_cfg = DefaultTrainConfig()

# prepare data
reader = DataReader(pascal_data_cfg)
roidb = reader.prepare_roidb()
roidb.filter_roidb(vgg_faster_rcnn_cfg)

# build model
model = VGGNetFasterRCNNModel(vgg_faster_rcnn_cfg)
predict = model.inference(model.TrainEnd2End)
# result = model.inference(model.Evaluation)

# setup trainer
trainer = Trainer(default_train_cfg)

writer = tf.summary.FileWriter('./tensorboard/cat')

loss = trainer.loss()
train_op = trainer.train(loss)

max_iters = 40000
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer.add_graph(sess.graph)

    model.load_weights_pre_trained(sess)

    merged = tf.summary.merge_all()

    for i in range(max_iters):

        blob = roidb.forward()
        feed = {model.data: blob['data'], model.gt_boxes: blob['gt_boxes'],
                model.im_info: blob['im_info'], model.keep_prob: 0.5}

        # pred = [predict[0].eval(sess), predict[1].eval(sess), predict[2].eval(sess)]
        # precision, accuracy = trainer.accuracy(pred, blob)
        # tf.summary.image('ori_image', blob['data'])

        # feed_eval = {
        #     trainer.data: blob['data'], trainer.gt_boxes: blob['gt_boxes'],
        #     trainer.im_info: blob['im_info']}
        #
        # precision, accuracy = sess.run([eval_op], feed_dict=feed_eval)
        train_loss, train_pred, _, summary = sess.run([loss, predict, train_op, merged], feed_dict=feed)

        # precision, accuracy = trainer.accuracy(train_pred, blob)

        # summary = sess.run([merged], feed_dict=feed)
        # tf.summary.scalar('precision', precision)

        # summary = sess.run(merged)

        # scores = train_pred[0]
        # bbox_deltas = train_pred[1]
        # rois = train_pred[2][0]
        #
        # scale = blob['im_info'][0][-1]
        # boxes = rois[:, 1:5]
        #
        # pred_bboxes = bbox_transform_inv(boxes, bbox_deltas)
        # pred_bboxes = clip_boxes(pred_bboxes, blob['im_info'][0])
        #
        # overlaps = bbox_overlaps(
        #     np.ascontiguousarray(blob['gt_boxes'][:, :4], dtype=np.float32),
        #     np.ascontiguousarray(pred_bboxes, dtype=np.float32))
        # max_overlaps = overlaps.max(axis=1)
        # ind_overlaps = overlaps.argmax(axis=1)
        # inds = np.where(max_overlaps >= 0.5)[0]
        # if len(inds) == 0:
        #     precision = 0.0
        #     accuracy = 0.0
        # else:
        #     positive = blob['gt_boxes'][:, 4][inds]
        #     cls_pred = np.argmax(scores[ind_overlaps[inds]], axis=1)
        #     true_pos = (positive == cls_pred)
        #     precision = np.sum(true_pos) / len(positive)
        #
        #     cls_prob = np.max(scores[ind_overlaps[inds]], axis=1)
        #     prob_list = []
        #     for idx in range(len(true_pos)):
        #         if bool(true_pos[idx]) is True:
        #             prob_list.append(cls_prob[idx])
        #         else:
        #             prob_list.append(0)
        #     accuracy = np.array(prob_list).mean()

        # print('iters: {}, get loss: {:1f}, precision: {:1f}, accuracy: {:1f}'.format(
        #     i, train_loss, precision, accuracy))
        print('iters: {}, get loss: {:1f}'.format(i, train_loss))

        writer.add_summary(summary, i)

        if i % 500 == 0 and i > 0:
            saver.save(sess, './tensorboard/cat/ckpt/cat_iter_' + str(i) + '.ckpt')
            print('save snapshot success !')
        # mesh_x = sess.run(mx)
        # print(mesh_x.shape)

# model.inference(blob['data'], blob['gt_boxes'], blob['im_info'])

print('done')
