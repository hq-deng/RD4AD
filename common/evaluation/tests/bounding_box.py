import sys
import unittest

import numpy as np

sys.path.insert(0, '..')
sys.path.insert(0, "../../..")

from bounding_box import BoundingBoxMetrics, GTKind, PredKind

class TestBoundingBoxMetrics(unittest.TestCase):
    def test_perfect_score_large(self):
        score = BoundingBoxMetrics(8, 8)

        score.update(
            {
                'height': 20,
                'width': 20,
                'bboxes': [
                    {
                        'left': 0,
                        'top': 0,
                        'width': 0.25,
                        'height': 0.25,
                    },
                    {
                        'left': 0.25,
                        'top': 0.5,
                        'width': 0.5,
                        'height': 0.25,
                    },
                ]
            },
            np.array([
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            pred_kind=PredKind.BBOX_DICT,
            gt_kind=GTKind.MASK,
            confs_pred=[0.50, 0.50]
        )
        result = score.compute()
        self.assertEqual(result["map_results"]["map"], 1.)
        self.assertEqual(result["image_classification_f1"], 1.)
        self.assertEqual(result["image_classification_accuracy"], 1.)

    def test_perfect_score_small(self):
        score = BoundingBoxMetrics(4, 4)

        score.update(
            {
                'height': 20,
                'width': 20,
                'bboxes': [
                    {
                        'left': 0,
                        'top': 0,
                        'width': 0.25,
                        'height': 0.25,
                    },
                    {
                        'left': 0.25,
                        'top': 0.5,
                        'width': 0.5,
                        'height': 0.25,
                    },
                ]
            },
            np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]),
            pred_kind=PredKind.BBOX_DICT,
            gt_kind=GTKind.MASK,
            confs_pred=[0.50, 0.50]
        )
        result = score.compute()
        self.assertEqual(result["map_results"]["map"], 1.)
        self.assertEqual(result["image_classification_f1"], 1.)
        self.assertEqual(result["image_classification_accuracy"], 1.)

    def test_half_score_small(self):
        """
        Precision-recall curve is a flat line at 0.5.
        """
        score = BoundingBoxMetrics(4, 4)

        score.update(
            {
                'height': 20,
                'width': 20,
                'bboxes': [
                    # False positive
                    {
                        'left': 0,
                        'top': 0,
                        'width': 0.25,
                        'height': 0.25,
                    },
                    # True positive with IoU of 1
                    {
                        'left': 0.25,
                        'top': 0.5,
                        'width': 0.5,
                        'height': 0.25,
                    },
                ]
            },
            np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]),
            pred_kind=PredKind.BBOX_DICT,
            gt_kind=GTKind.MASK,
            confs_pred=[0.50, 0.50]
        )
        result = score.compute()
        self.assertEqual(result["map_results"]["map"], 0.5)
        self.assertEqual(result["image_classification_f1"], 1.)
        self.assertEqual(result["image_classification_accuracy"], 1.)

    def test_perfect_score_large_heatmap(self):
        score = BoundingBoxMetrics(8, 8)

        score.update(
            np.array([
                [0.3, 0.2, 0, 0, 0, 0, 0, 0],
                [0.2, 0.3, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.3, 0.5, 0.3, 0.5, 0, 0],
                [0, 0, 0.5, 0.3, 0.5, 0.3, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            np.array([
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            pred_kind=PredKind.HEATMAP,
            gt_kind=GTKind.MASK,
            pred_threshold=0.1,
        )
        result = score.compute()
        self.assertEqual(result["map_results"]["map"], 1.)
        self.assertEqual(result["image_classification_f1"], 1.)
        self.assertEqual(result["image_classification_accuracy"], 1.)

    def test_perfect_score_small_heatmap(self):
        score = BoundingBoxMetrics(4, 4)

        score.update(
            np.array([
                [0.2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0.5, 0.5, 0],
                [0, 0, 0, 0],
            ]),
            np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]),
            pred_kind=PredKind.HEATMAP,
            gt_kind=GTKind.MASK,
            pred_threshold=0.1,
        )
        result = score.compute()
        self.assertEqual(result["map_results"]["map"], 1.)
        self.assertEqual(result["image_classification_f1"], 1.)
        self.assertEqual(result["image_classification_accuracy"], 1.)

    def test_half_score_small_heatmap(self):
        """
        Precision-recall curve is a flat line at 0.5 precision.
        """
        score = BoundingBoxMetrics(4, 4)

        score.update(
            np.array([
                [0.8, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0.2, 0.3, 0],
                [0, 0, 0, 0],
            ]),
            np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]),
            pred_kind=PredKind.HEATMAP,
            gt_kind=GTKind.MASK,
            pred_threshold=0.1,
        )
        result = score.compute()
        self.assertEqual(result["map_results"]["map"], 0.5)
        self.assertEqual(result["image_classification_f1"], 1.)
        self.assertEqual(result["image_classification_accuracy"], 1.)

    def test_half_score_small_heatmap_2(self):
        """
        Precision-recall curve is a vertical line at 0.5 recall.
        """
        # There is ambiguity in how 0.5 is handled, so we can explicitly avoid that threshold
        score = BoundingBoxMetrics(
            4, 4, rec_thresholds=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        )

        score.update(
            np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0.8, 0.6, 0],
                [0, 0, 0, 0],
            ]),
            np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]),
            pred_kind=PredKind.HEATMAP,
            gt_kind=GTKind.MASK,
            pred_threshold=0.1,
        )
        result = score.compute()
        self.assertEqual(result["map_results"]["map"], 0.5)
        self.assertEqual(result["image_classification_f1"], 1.)
        self.assertEqual(result["image_classification_accuracy"], 1.)

    # Test edge case where bboxes is 0
    def test_edge_case_where_bboxes_is_empty(self):
        score = BoundingBoxMetrics(8, 8)

        score.update(
            {
                'height': 20,
                'width': 20,
                'bboxes': [],
            },
            np.array([
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            pred_kind=PredKind.BBOX_DICT,
            gt_kind=GTKind.MASK,
            confs_pred=[]
        )
        result = score.compute()
        self.assertEqual(result["map_results"]["map"], 0.)
        self.assertEqual(result["image_classification_f1"], 0.)
        self.assertEqual(result["image_classification_accuracy"], 0.)

    def test_resizing(self):
        """
        Tests the case where the bounding box dictionary uses heights
        and widths that do not match the provided ground-truth mask.
        This is useful for cases when dictionaries of bounding boxes
        produced by Matroid detectors use a non-square-padded image, whereas
        the ground-truth produced by the client code uses a padded image.
        """
        score = BoundingBoxMetrics(8, 8)

        score.update(
            {
                'height': 10,
                'width': 20,
                'bboxes': [
                    {
                        'left': 0.25,
                        'top': 0.5,
                        'width': 0.5,
                        'height': 0.5,
                    },
                ]
            },
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            pred_kind=PredKind.BBOX_DICT,
            gt_kind=GTKind.MASK,
            confs_pred=[0.50]
        )
        result = score.compute()
        self.assertEqual(result["map_results"]["map"], 1.)
        self.assertEqual(result["image_classification_f1"], 1.)
        self.assertEqual(result["image_classification_accuracy"], 1.)

    def test_stress(self):
        score = BoundingBoxMetrics(8, 8, rec_thresholds=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        # Image 1
        score.update(
            {
                'height': 10,
                'width': 20,
                'bboxes': [
                    {
                        'left': 0.25,
                        'top': 0.5,
                        'width': 0.5,
                        'height': 0.5,
                    },
                ]
            },
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            pred_kind=PredKind.BBOX_DICT,
            gt_kind=GTKind.MASK,
            confs_pred=[0.50]
        )
        # Image 2
        score.update(
            {
                'height': 20,
                'width': 10,
                # 75% overlap with gt
                'bboxes': [
                    {
                        'left': 0.0,
                        'top': 0.5,
                        'width': 0.75,
                        'height': 0.25,
                    },
                ]
            },
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            pred_kind=PredKind.BBOX_DICT,
            gt_kind=GTKind.MASK,
            confs_pred=[0.80]
        )
        # Image 3
        score.update(
            {
                'height': 40,
                'width': 40,
                # Two false positives
                'bboxes': [
                    {
                        'left': 0.0,
                        'top': 0.5,
                        'width': 0.75,
                        'height': 0.25,
                    },
                    # 5x5 pixels near the bottom
                    # right corner
                    {
                        'left': 0.8,
                        'top': 0.8,
                        'width': 0.125,
                        'height': 0.125,
                    },
                ]
            },
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            pred_kind=PredKind.BBOX_DICT,
            gt_kind=GTKind.MASK,
            confs_pred=[0.81, 0.90]
        )
        # Image 4
        score.update(
            {
                'height': 30,
                'width': 30,
                # Two false negative. One false
                # positive. False positive has
                # a slight overlap, but the overlap
                # is too small
                'bboxes': [
                    {
                        'left': 0.0,
                        'top': 0.0,
                        'width': 0.5,
                        'height': 0.5,
                    },
                    {
                        'left': 0.75,
                        'top': 0.75,
                        'width': 0.25,
                        'height': 0.25,
                    },
                ]
            },
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1],
            ]),
            pred_kind=PredKind.BBOX_DICT,
            gt_kind=GTKind.MASK,
            confs_pred=[0.2, 1.0]
        )
        result = score.compute()
        self.assertEqual(round(result["map_results"]["map_50"].item(), 4), round(6/11, 4))
        self.assertEqual(result["image_classification_f1"], 6/7)
        self.assertEqual(result["image_classification_accuracy"], 0.75)

    def test_classification_scores(self):
        """
        Tests whether the whole-image classification metrics are correct.
        """
        # Checks the edge case where the image height and width we pass in
        # are different from the actual masks
        score = BoundingBoxMetrics(1, 5)

        score.update(
            np.array([[1, 0], [0, 0]]),
            np.array([[1, 0], [0, 0]]),
            pred_kind=PredKind.HEATMAP,
            gt_kind=GTKind.MASK,
            pred_threshold=0.1,
        )
        score.update(
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 0], [0, 0]]),
            pred_kind=PredKind.HEATMAP,
            gt_kind=GTKind.MASK,
            pred_threshold=0.1,
        )
        score.update(
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 0]]),
            pred_kind=PredKind.HEATMAP,
            gt_kind=GTKind.MASK,
            pred_threshold=0.1,
        )
        score.update(
            np.array([[1, 0], [0, 0]]),
            np.array([[1, 0], [0, 0]]),
            pred_kind=PredKind.HEATMAP,
            gt_kind=GTKind.MASK,
            pred_threshold=0.1,
        )
        score.update(
            np.array([[1, 0], [0, 0]]),
            np.array([[1, 0], [0, 0]]),
            pred_kind=PredKind.HEATMAP,
            gt_kind=GTKind.MASK,
            pred_threshold=0.1,
        )
        score.update(
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 0]]),
            pred_kind=PredKind.HEATMAP,
            gt_kind=GTKind.MASK,
            pred_threshold=0.1,
        )
        score.update(
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 0]]),
            pred_kind=PredKind.HEATMAP,
            gt_kind=GTKind.MASK,
            pred_threshold=0.1,
        )
        score.update(
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 0]]),
            pred_kind=PredKind.HEATMAP,
            gt_kind=GTKind.MASK,
            pred_threshold=0.1,
        )
        score.update(
            np.array([[0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0]]),
            pred_kind=PredKind.HEATMAP,
            gt_kind=GTKind.MASK,
            pred_threshold=0.1,
        )

        recall = 3/4
        precision = 3/7
        accuracy = 4/9

        result = score.compute()
        self.assertEqual(result["image_classification_f1"], 2*precision*recall/(precision + recall))
        self.assertEqual(result["image_classification_accuracy"], accuracy)


if __name__ == '__main__':
    unittest.main()
