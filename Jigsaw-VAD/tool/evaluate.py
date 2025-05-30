import json
import math
from sklearn import metrics
import pickle
import argparse
import os
import numpy as np
from numpy.core.fromnumeric import nonzero
import scipy.io as scio
import sys
from sklearn.utils.validation import check_non_negative

from torch import dist
sys.path.append('../')
DATA_DIR = os.environ["VAD_DATASET_PATH"]

# normalize scores in each sub video
NORMALIZE = True

# number of history frames, since in prediction based method, the first 4 frames can not be predicted, so that
# the first 4frames are undecidable, we just ignore the first 4 frames
DECIDABLE_IDX = 3


def parser_args():
    parser = argparse.ArgumentParser(description='evaluating the model, computing the roc/auc.')

    parser.add_argument('-f', '--file', type=str, help='the path of loss file.')
    parser.add_argument('-t', '--type', type=str, default='compute_auc',
                        help='the type of evaluation, choosing type is: plot_roc, compute_auc, '
                             'test_func\n, the default type is compute_auc')
    return parser.parse_args()


def score_smoothing(score, ws=25, function='mean', sigma=10):
    assert ws % 2 == 1, 'window size must be odd'
    assert function in ['mean', 'gaussian'], 'wrong type of window function'

    r = ws // 2
    weight = np.ones(ws)
    for i in range(ws):
        if function == 'mean':
            weight[i] = 1. / ws
        elif function == 'gaussian':
            weight[i] = np.exp(-(i - r) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    weight /= weight.sum()
    new_score = score.copy()
    new_score[r: score.shape[0] - r] = np.correlate(score, weight, mode='valid')
    return new_score


def gaussian_filter_(support, sigma):
    mu = support[len(support) // 2 - 1]
    filter = 1.0 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filter


class RecordResult(object):
    def __init__(self, fpr=None, tpr=None, auc=-np.inf, dataset=None, thresholds=None, labels=None, scores=None):
        self.fpr = fpr
        self.tpr = tpr
        self.auc = auc
        self.ths = thresholds
        self.labels = labels
        self.scores = scores
        self.dataset = dataset

    def get_accuracy_at_call(self, threshold_FN):
        threshold = self.ths[self.tpr > (1-threshold_FN)][0]
        pred = np.where(self.scores > threshold, 1, 0)
        return (pred == self.labels)[pred == 1].mean()

    def get_AP_curve(self):
        return metrics.precision_recall_curve(self.labels, self.scores)

    def get_AP_score(self):
        return metrics.average_precision_score(self.labels, self.scores)

    def __lt__(self, other):
        return self.auc < other.auc

    def __gt__(self, other):
        return self.auc > other.auc

    def __str__(self):
        return 'dataset = {}, auc = {}'.format(self.dataset, self.auc)


class GroundTruthLoader(object):
    AVENUE = 'avenue'
    PED1 = 'ped1'
    PED1_PIXEL_SUBSET = 'ped1_pixel_subset'
    PED2 = 'ped2'
    ENTRANCE = 'enter'
    EXIT = 'exit'
    SHANGHAITECH = 'shanghaitech'
    SHANGHAITECH_LABEL_PATH = os.path.join(DATA_DIR, 'shanghaitech/frame_masks')
    TOY_DATA = 'toydata'
    TOY_DATA_LABEL_PATH = os.path.join(DATA_DIR, TOY_DATA, 'toydata.json')

    NAME_MAT_MAPPING = {
        AVENUE: os.path.join(DATA_DIR, 'avenue_frames/avenue.mat'),
        PED1: os.path.join(DATA_DIR, 'ped1/ped1.mat'),
        PED2: os.path.join(DATA_DIR, 'ped2/ped2.mat'),
        ENTRANCE: os.path.join(DATA_DIR, 'enter/enter.mat'),
        EXIT: os.path.join(DATA_DIR, 'exit/exit.mat')
    }

    NAME_FRAMES_MAPPING = {
        AVENUE: os.path.join(DATA_DIR, 'avenue_frames/testing'),
        PED1: os.path.join(DATA_DIR, 'ped1/testing'),
        PED2: os.path.join(DATA_DIR, 'ped2/testing'),
        ENTRANCE: os.path.join(DATA_DIR, 'enter/testing/frames'),
        EXIT: os.path.join(DATA_DIR, 'exit/testing/frames')
    }

    def __init__(self, mapping_json=None):
        """
        Initial a ground truth loader, which loads the ground truth with given dataset name.

        :param mapping_json: the mapping from dataset name to the path of ground truth.
        """

        if mapping_json is not None:
            with open(mapping_json, 'rb') as json_file:
                self.mapping = json.load(json_file)
        else:
            self.mapping = GroundTruthLoader.NAME_MAT_MAPPING

    def __call__(self, dataset):
        """ get the ground truth by provided the name of dataset.

        :type dataset: str
        :param dataset: the name of dataset.
        :return: np.ndarray, shape(#video)
                 np.array[0] contains all the start frame and end frame of abnormal events of video 0,
                 and its shape is (#frapsnr, )
        """

        if dataset == GroundTruthLoader.SHANGHAITECH:
            gt = self.__load_shanghaitech_gt()
        elif dataset == GroundTruthLoader.TOY_DATA:
            gt = self.__load_toydata_gt()
        elif dataset == GroundTruthLoader.PED2:
            gt = self.__load_ucsd_gt(dataset)
        elif dataset == GroundTruthLoader.AVENUE:
            gt = self.__load_avenue_gt(dataset)
        return gt

    # def __load_ucsd_avenue_subway_gt(self, dataset):
    #     assert dataset in self.mapping, 'there is no dataset named {} \n Please check {}' \
    #         .format(dataset, GroundTruthLoader.NAME_MAT_MAPPING.keys())

    #     mat_file = self.mapping[dataset]
    #     abnormal_events = scio.loadmat(mat_file, squeeze_me=True)['gt']

    #     if abnormal_events.ndim == 2:
    #         abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0], abnormal_events.shape[1])

    #     num_video = abnormal_events.shape[0]
    #     dataset_video_folder = GroundTruthLoader.NAME_FRAMES_MAPPING[dataset]
    #     video_list = os.listdir(dataset_video_folder)
    #     video_list.sort()

    #     assert num_video == len(video_list), 'ground true does not match the number of testing videos. {} != {}' \
    #         .format(num_video, len(video_list))

    #     # get the total frames of sub video
    #     def get_video_length(sub_video_number):
    #         # video_name = video_name_template.format(sub_video_number)
    #         video_name = os.path.join(dataset_video_folder, video_list[sub_video_number])
    #         assert os.path.isdir(video_name), '{} is not directory!'.format(video_name)

    #         length = len(os.listdir(video_name))

    #         return length

    #     # need to test [].append, or np.array().append(), which one is faster
    #     gt = []
    #     for i in range(num_video):
    #         length = get_video_length(i)

    #         sub_video_gt = np.zeros((length,), dtype=np.int8)
    #         sub_abnormal_events = abnormal_events[i]
    #         if sub_abnormal_events.ndim == 1:
    #             sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))

    #         _, num_abnormal = sub_abnormal_events.shape

    #         for j in range(num_abnormal):
    #             # (start - 1, end - 1)
    #             start = sub_abnormal_events[0, j] - 1
    #             end = sub_abnormal_events[1, j]

    #             sub_video_gt[start: end] = 1

    #         gt.append(sub_video_gt)

    #     return gt

    def __load_ucsd_gt(self, dataset):
        assert dataset in self.mapping, 'there is no dataset named {} \n Please check {}' \
            .format(dataset, GroundTruthLoader.NAME_MAT_MAPPING.keys())

        mat_file = self.mapping[dataset]
        abnormal_events = scio.loadmat(mat_file, squeeze_me=True)['gt']

        if abnormal_events.ndim == 2:
            abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0], abnormal_events.shape[1])

        num_video = abnormal_events.shape[0]
        dataset_video_folder = GroundTruthLoader.NAME_FRAMES_MAPPING[dataset]
        video_list = os.listdir(dataset_video_folder)
        video_list.sort()

        assert num_video == len(video_list), 'ground true does not match the number of testing videos. {} != {}' \
            .format(num_video, len(video_list))

        # get the total frames of sub video
        def get_video_length(sub_video_number):
            # video_name = video_name_template.format(sub_video_number)
            video_name = os.path.join(dataset_video_folder, video_list[sub_video_number])
            assert os.path.isdir(video_name), '{} is not directory!'.format(video_name)

            length = len(os.listdir(video_name))

            return length

        # need to test [].append, or np.array().append(), which one is faster
        gt = []
        for i in range(num_video):
            length = get_video_length(i)

            sub_video_gt = np.zeros((length,), dtype=np.int8)
            sub_abnormal_events = abnormal_events[i]
            if sub_abnormal_events.ndim == 1:
                sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))

            _, num_abnormal = sub_abnormal_events.shape

            for j in range(num_abnormal):
                # (start - 1, end - 1)
                start = sub_abnormal_events[0, j] - 1
                end = sub_abnormal_events[1, j]

                sub_video_gt[start: end] = 1

            gt.append(sub_video_gt)

        return gt

    def __load_avenue_gt(self, dataset):
        root = f'{DATA_DIR}{dataset}_testing_label_mask'
        assert dataset in self.mapping, 'there is no dataset named {} \n Please check {}' \
            .format(dataset, GroundTruthLoader.NAME_MAT_MAPPING.keys())
        if os.path.exists(os.path.join(root, 'label.pkl')):
            with open(os.path.join(root, 'label.pkl'), 'rb') as f:
                gt = pickle.load(f)
            return gt
        num_video = 21
        dataset_video_folder = GroundTruthLoader.NAME_FRAMES_MAPPING[dataset]
        video_list = [os.path.splitext(file)[0] for file in os.listdir(dataset_video_folder)]
        video_list.sort()

        assert num_video == len(video_list), 'ground true does not match the number of testing videos. {} != {}' \
            .format(num_video, len(video_list))

        # need to test [].append, or np.array().append(), which one is faster
        gt = []
        for i in range(num_video):
            mat_path = os.path.join(root, str(int(video_list[i])) + '_label.mat')
            sub_video_gt = np.stack(scio.loadmat(mat_path)['volLabel'][0]).max((1, 2))
            gt.append(sub_video_gt)

        with open(os.path.join(root, 'label.pkl'), 'wb') as f:
            pickle.dump(gt, f)
        return gt

    @staticmethod
    def __load_shanghaitech_gt():
        video_path_list = os.listdir(GroundTruthLoader.SHANGHAITECH_LABEL_PATH)
        video_path_list.sort()
        video_path_list = video_path_list

        gt = []
        for video in video_path_list:
            gt.append(np.load(os.path.join(GroundTruthLoader.SHANGHAITECH_LABEL_PATH, video)))

        return gt

    @staticmethod
    def __load_toydata_gt():
        with open(GroundTruthLoader.TOY_DATA_LABEL_PATH, 'r') as gt_file:
            gt_dict = json.load(gt_file)

        gt = []
        for video, video_info in gt_dict.items():
            length = video_info['length']
            video_gt = np.zeros((length,), dtype=np.uint8)
            sub_gt = np.array(np.matrix(video_info['gt']))

            for anomaly in sub_gt:
                start = anomaly[0]
                end = anomaly[1] + 1
                video_gt[start: end] = 1
            gt.append(video_gt)
        return gt

    @staticmethod
    def get_pixel_masks_file_list(dataset):
        # pixel mask folder
        pixel_mask_folder = os.path.join(DATA_DIR, f'{dataset}_frames', 'pixel_masks')
        pixel_mask_file_list = os.listdir(pixel_mask_folder)
        pixel_mask_file_list.sort()

        # get all testing videos
        dataset_video_folder = GroundTruthLoader.NAME_FRAMES_MAPPING[dataset]
        video_list = os.listdir(dataset_video_folder)
        video_list.sort()

        # get all testing video names with pixel masks
        pixel_video_ids = []
        ids = 0
        for pixel_mask_name in pixel_mask_file_list:
            while ids < len(video_list):
                if video_list[ids] + '.npy' == pixel_mask_name:
                    pixel_video_ids.append(ids)
                    ids += 1
                    break
                else:
                    ids += 1

        assert len(pixel_video_ids) == len(pixel_mask_file_list)

        for i in range(len(pixel_mask_file_list)):
            pixel_mask_file_list[i] = os.path.join(pixel_mask_folder, pixel_mask_file_list[i])

        return pixel_mask_file_list, pixel_video_ids


def load_psnr_gt(results):
    # with open(loss_file, 'rb') as reader:
    #     # results {
    #     #   'dataset': the name of dataset
    #     #   'psnr': the psnr of each testing videos,
    #     # }

    #     # psnr_records['psnr'] is np.array, shape(#videos)
    #     # psnr_records[0] is np.array   ------>     01.avi
    #     # psnr_records[1] is np.array   ------>     02.avi
    #     #               ......
    #     # psnr_records[n] is np.array   ------>     xx.avi

    #     results = pickle.load(reader)
    # with open(results, 'rb') as f:
    #     results = pickle.load(f)

    dataset = results['dataset']
    psnr_records = results['psnr']

    num_videos = len(psnr_records)

    # load ground truth
    gt_loader = GroundTruthLoader()
    gt = gt_loader(dataset=dataset)

    assert num_videos == len(gt), 'the number of saved videos does not match the ground truth, {} != {}' \
        .format(num_videos, len(gt))

    return dataset, psnr_records, gt


def load_psnr_gt_flow(loss_file):
    with open(loss_file, 'rb') as reader:
        # results {
        #   'dataset': the name of dataset
        #   'psnr': the psnr of each testing videos,
        # }

        # psnr_records['psnr'] is np.array, shape(#videos)
        # psnr_records[0] is np.array   ------>     01.avi
        # psnr_records[1] is np.array   ------>     02.avi
        #               ......
        # psnr_records[n] is np.array   ------>     xx.avi

        results = pickle.load(reader)

    dataset = results['dataset']
    psnrs = results['psnr']
    flows = results['flow']

    num_videos = len(psnrs)

    # load ground truth
    gt_loader = GroundTruthLoader()
    gt = gt_loader(dataset=dataset)

    assert num_videos == len(gt), 'the number of saved videos does not match the ground truth, {} != {}' \
        .format(num_videos, len(gt))

    return dataset, psnrs, flows, gt


def load_psnr(loss_file):
    """
    load image psnr or optical flow psnr.
    :param loss_file: loss file path
    :return:
    """
    with open(loss_file, 'rb') as reader:
        # results {
        #   'dataset': the name of dataset
        #   'psnr': the psnr of each testing videos,
        # }

        # psnr_records['psnr'] is np.array, shape(#videos)
        # psnr_records[0] is np.array   ------>     01.avi
        # psnr_records[1] is np.array   ------>     02.avi
        #               ......
        # psnr_records[n] is np.array   ------>     xx.avi

        results = pickle.load(reader)
    psnrs = results['psnr']
    return psnrs


def get_scores_labels(loss_file, reverse, smoothing):
    # the name of dataset, loss, and ground truth
    dataset, psnr_records, gt = load_psnr_gt(loss_file=loss_file)

    # the number of videos
    num_videos = len(psnr_records)

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    # video normalization
    for i in range(num_videos):
        distance = psnr_records[i]

        if NORMALIZE:
            distance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-8)
            # distance -= distance.min()  # distances = (distance - min) / (max - min)
            # distance /= distance.max()
            if reverse:
                distance = 1 - distance
        if smoothing:
            distance = score_smoothing(distance)
        scores = np.concatenate((scores[:], distance[DECIDABLE_IDX:-DECIDABLE_IDX]), axis=0)
        labels = np.concatenate((labels[:], gt[i][DECIDABLE_IDX:-DECIDABLE_IDX]), axis=0)
    return dataset, scores, labels


def precision_recall_auc(loss_file, reverse, smoothing):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        dataset, scores, labels = get_scores_labels(sub_loss_file, reverse, smoothing)
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=0)
        auc = metrics.auc(recall, precision)

        results = RecordResult(recall, precision, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model PR-AUC = {}'.format(optimal_results))
    return optimal_results


def cal_eer(fpr, tpr):
    # makes fpr + tpr = 1
    eer = fpr[np.nanargmin(np.absolute((fpr + tpr - 1)))]
    return eer


def compute_eer(loss_file, reverse, smoothing):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult(auc=np.inf)
    for sub_loss_file in loss_file_list:
        dataset, scores, labels = get_scores_labels(sub_loss_file, reverse, smoothing)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        eer = cal_eer(fpr, tpr)

        results = RecordResult(fpr, tpr, eer, dataset, sub_loss_file)

        if optimal_results > results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model EER = {}'.format(optimal_results))
    return optimal_results


def compute_auc(res, reverse, smoothing):
    dataset, psnr_records, gt = load_psnr_gt(res)
    # if dataset=='shanghaitech':
    #     gt[51][5]=0

    num_videos = len(psnr_records)
    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    for i in range(num_videos):
        distance = psnr_records[i]
        if NORMALIZE:
            distance = distance
            # distance -= distance.min() + 1e-8
            # distance /= distance.max()
            if reverse:
                distance = 1 - distance
        # smooth the score
        if smoothing:
            filter_2d = gaussian_filter_(np.arange(1, 50), 20)
            padding_size = len(filter_2d) // 2
            in_ = np.concatenate((distance[:padding_size], distance, distance[-padding_size:]))
            distance = np.correlate(in_, filter_2d, 'valid')
            # distance = score_smoothing(distance)
        scores = np.concatenate((scores[:], distance), axis=0)
        labels = np.concatenate((labels[:], gt[i]), axis=0)

    # print("label.shape:{}, scores.shape:{}".format(labels.shape, scores.shape))
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    results = RecordResult(fpr, tpr, auc, dataset, thresholds, labels, scores)
    return results


def compute_auc_average(res, reverse, smoothing):
    auc_list = []
    dataset, psnr_records, gt = load_psnr_gt(res)
    file_name_dir = f'{DATA_DIR}{dataset}_frames/testing'
    file_name = os.listdir(file_name_dir)
    file_name.sort()

    # the number of videos
    num_videos = len(psnr_records)
    auc = 0
    for i in range(num_videos):
        distance = psnr_records[i]
        if NORMALIZE:
            # distance = (distance-distance.min())/(distance.max()-distance.min()+1e-8)
            if reverse:
                distance = 1 - distance
        # to smooth the score
        if smoothing:
            # distance = score_smoothing(distance)
            filter_2d = gaussian_filter_(np.arange(1, 50), 20)
            padding_size = len(filter_2d) // 2
            in_ = np.concatenate((distance[:padding_size], distance, distance[-padding_size:]))
            distance = np.correlate(in_, filter_2d, 'valid')

        # fpr, tpr, thresholds = metrics.roc_curve(np.array(gt[i],dtype=np.int8), np.array(distance,dtype=np.float32), pos_label=1)
        fpr, tpr, _ = metrics.roc_curve(
            np.concatenate(([0], np.array(gt[i], dtype=np.int8), [1])),
            np.concatenate(([0], np.array(distance, dtype=np.float32), [1])),
            pos_label=1)

        _auc = metrics.auc(fpr, tpr)
        # print('video {}: auc is {}'.format(file_name[i], _auc))
        auc += _auc
    auc /= num_videos
    auc_list.append(auc)
    return auc_list


def average_psnr(loss_file, reverse):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    max_avg_psnr = -np.inf
    max_file = ''
    for file in loss_file_list:
        psnr_records = load_psnr(file)

        psnr_records = np.concatenate(psnr_records, axis=0)
        avg_psnr = np.mean(psnr_records)
        if max_avg_psnr < avg_psnr:
            max_avg_psnr = avg_psnr
            max_file = file
        print('{}, average psnr = {}'.format(file, avg_psnr))

    print('max average psnr file Averge Score = {}, psnr = {}'.format(max_file, max_avg_psnr))


def calculate_psnr(loss_file, reverse, smoothing):
    optical_result = compute_auc(loss_file, reverse, smoothing)
    print('##### optimal result and model = {}'.format(optical_result))

    mean_psnr = []
    for file in os.listdir(loss_file):
        file = os.path.join(loss_file, file)
        dataset, psnr_records, gt = load_psnr_gt(file)

        psnr_records = np.concatenate(psnr_records, axis=0)
        gt = np.concatenate(gt, axis=0)

        mean_normal_psnr = np.mean(psnr_records[gt == 0])
        mean_abnormal_psnr = np.mean(psnr_records[gt == 1])
        mean = np.mean(psnr_records)
        print('mean normal psrn = {}, mean abnormal psrn = {}, mean = {}'.format(
            mean_normal_psnr,
            mean_abnormal_psnr,
            mean)
        )
        mean_psnr.append(mean)
    print('max mean psnr = {}'.format(np.max(mean_psnr)))


def calculate_score(loss_file, reverse, smoothing):
    if not os.path.isdir(loss_file):
        loss_file_path = loss_file
    else:
        optical_result = compute_auc(loss_file, reverse, smoothing)
        loss_file_path = optical_result.loss_file
        print('##### optimal result and model = {}'.format(optical_result))
    dataset, psnr_records, gt = load_psnr_gt(loss_file=loss_file_path)

    # the number of videos
    num_videos = len(psnr_records)

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    # video normalization
    for i in range(num_videos):
        distance = psnr_records[i]

        distance = (distance - distance.min()) / (distance.max() - distance.min())
        if reverse:
            distance = 1-distance
        if smoothing:
            distance = score_smoothing(distance)
        scores = np.concatenate((scores[:], distance[DECIDABLE_IDX:-DECIDABLE_IDX]), axis=0)
        labels = np.concatenate((labels[:], gt[i][DECIDABLE_IDX:-DECIDABLE_IDX]), axis=0)

    mean_normal_scores = np.mean(scores[labels == 0])
    mean_abnormal_scores = np.mean(scores[labels == 1])
    print('mean normal scores = {}, mean abnormal scores = {}, '
          'delta = {}'.format(mean_normal_scores, mean_abnormal_scores, mean_normal_scores - mean_abnormal_scores))


def test_func(*args):
    # simulate testing on CUHK AVENUE dataset
    dataset = GroundTruthLoader.AVENUE

    # load the ground truth
    gt_loader = GroundTruthLoader()
    gt = gt_loader(dataset=dataset)

    num_videos = len(gt)

    simulated_results = {
        'dataset': dataset,
        'psnr': []
    }

    simulated_psnr = []
    for i in range(num_videos):
        sub_video_length = gt[i].shape[0]
        simulated_psnr.append(np.random.random(size=sub_video_length))

    simulated_results['psnr'] = simulated_psnr

    # writing to file, 'generated_loss.bin'
    with open('generated_loss.bin', 'wb') as writer:
        pickle.dump(simulated_results, writer, pickle.HIGHEST_PROTOCOL)

    print(file_path.name)
    result = compute_auc(file_path.name)

    print('optimal = {}'.format(result))


eval_type_function = {
    'compute_auc': compute_auc,
    'compute_eer': compute_eer,
    'precision_recall_auc': precision_recall_auc,
    'calculate_psnr': calculate_psnr,
    'calculate_score': calculate_score,
    'average_psnr': average_psnr,
    'average_psnr_sample': average_psnr
}


def evaluate(eval_type, save_file):
    assert eval_type in eval_type_function, 'there is no type of evaluation {}, please check {}' \
        .format(eval_type, eval_type_function.keys())
    eval_func = eval_type_function[eval_type]
    optimal_results = eval_func(save_file)
    return optimal_results


def evaluate_all(res, reverse=True, smoothing=True):
    result = compute_auc(res, reverse, smoothing)
    aver_result = compute_auc_average(res, reverse, smoothing)
    return result, aver_result


if __name__ == '__main__':
    pickle_path = './test.pkl'
    result = evaluate_all(pickle_path, reverse=True, smoothing=True)
    print(result)
