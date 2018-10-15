import argparse
import sys

def argparser():
    parser = argparse.ArgumentParser()
    # for model
    parser.add_argument(
        '--filter_sizes',
        default = [8, 12, 16, 20, 24, 28, 32, 36],
        type=int,
        nargs='+',
        help='Space seperated list of motif filter lengths. (ex, --filter_sizes 4 8 12)'
    )
    parser.add_argument(
        '--num_filters',
        default = [256, 256, 256, 256, 256, 256, 256, 256],
        type=int,
        nargs='+',
        help='Space seperated list of the number of convolution filters corresponding to length list. (ex, --num_filters 100 200 100)'
    )
    parser.add_argument(
        '--num_hidden',
        type=int,
        default=512,
        help='Number of neurons in hidden layer.'
    )
    parser.add_argument(
        '--regularizer',
        type=float,
        default=0.001,
        help='(Lambda value / 2) of L2 regularizer on weights connected to last layer (0 to exclude).'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Rate for dropout.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=1000,
        help='Number of classes (families).'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=1000,
        help='Length of input sequences.'
    )
    # for learning
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs to train.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--train_file',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for input data.'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for input data.'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='',
        help='Path to write checkpoint file.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp',
        help='Directory for log data.'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=100,
        help='Interval of steps for logging.'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=100,
        help='Interval of steps for save model.'
    )
    # test
    parser.add_argument(
        '--fine_tuning',
        type=bool,
        default=False,
        help='If true, weight on last layer will not be restored.'
    )
    parser.add_argument(
        '--fine_tuning_layers',
        type=str,
        nargs='+',
        default=["fc2"],
        help='Which layers should be restored. Default is ["fc2"].'
    )
    parser.add_argument(
        '--save_prediction',
        type=str,
        default=None,
        help='Path to save prediction'
    )

    try:
        FLAGS, unparsed = parser.parse_known_args()

    except:
        parser.print_help()
        sys.exit(1)

    # check validity
    assert (len(FLAGS.filter_sizes) == len(FLAGS.num_filters))

    return FLAGS