from __future__ import print_function
import argparse

__all__ = ['get_args']


def get_args():
    parser = argparse.ArgumentParser(description='RD4AD')
    parser.add_argument('--dataset', default='mvtec', type=str, metavar='D',
                        help='dataset name: mvtec/connectors/automotive (default: mvtec)')
    parser.add_argument('--checkpoint', default='', type=str, metavar='D',
                        help='file with saved checkpoint')
    parser.add_argument('--load-preds', action='store_true', default=False,
                        help='whether to load anomaly maps from a directory with \
                              the same name as the checkpoint path.')
    parser.add_argument('-cl', '--class-names', default=[None], nargs='+', type=str, metavar='C',
                        help='class names for MVTec/connectors/automotive (default: none)')
    parser.add_argument('-run', '--run-name', default=0, type=int, metavar='C',
                        help='name of the run (default: 0)')
    parser.add_argument('-inp', '--input-size', default=256, type=int, metavar='C',
                        help='image resize dimensions (default: 256)')
    parser.add_argument("--action-type", default='norm-train', type=str, metavar='T',
                        help=\
                            'norm-train/norm-test/norm-test-fps/norm-compare-to-matroid/ \
                             norm-patches-test/norm-patches-compare-to-matroid \
                             (default: norm-train)')
    parser.add_argument('-bs', '--batch-size', default=8, type=int, metavar='B',
                        help='train batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.0025, metavar='LR',
                        help='learning rate (default: 0.0025)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--viz', action='store_true', default=False,
                        help='saves test data visualizations')
    parser.add_argument("--gpu", default='0', type=str, metavar='G',
                        help='GPU device number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--multimodal', action='store_true', default=False,
                        help='whether to enable multimodal experiments on mvtec')
    parser.add_argument('--patches-per-row', default=0, type=int,
                        help='Number of patches to take from each row')
    parser.add_argument('--patches-per-column', default=0, type=int,
                        help='Number of patches to take from each column')
    parser.add_argument('--crop-width', default=None, type=int,
                        help='For BSD datasets only; cropping to apply AFTER the image is \
                              padded and resized.')
    parser.add_argument('--crop-height', default=None, type=int,
                        help='For BSD datasets only; cropping to apply AFTER the image is \
                              padded and resized.')
    parser.add_argument('--save-segmentation-images', action='store_true', default=False,
                        help='whether or not to save off example segmentation images')
    parser.add_argument('--train-split-percent', default=1.0, type=float,
                        help='the percentage of good/non-defective images to put in the train set.\
                              By default, all good images are put in the train set.')
    # TODO(@nraghuraman-matroid): Rather than requiring saved off images, use -cmp-bboxes
    # instead.
    parser.add_argument('-cmp', '--comparison-images', default='', type=str,
                        help='intended for comparison of visualized model segmentations \
                              with the segmentations of another model. If --save-segmentation-images \
                              is set and a path is provided, each visualization of a predicted map \
                              will be compared to a matching image at this path.')
    parser.add_argument('-cmp-bboxes', '--comparison-bboxes', default='', type=str,
                        help='A path to object detection bounding box predictions. Used only \
                              if --action-type is norm-compare-to-matroid'
                        )
    parser.add_argument("--pred-threshold", default=None, type=float, metavar='G',
                        help='For RD4AD anomaly heatmap predictions, the threshold beyond which \
                              to consider a pixel as anomalous.')

    args = parser.parse_args()
    
    return args
