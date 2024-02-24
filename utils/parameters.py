import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='The pytorch implementation for STTN'
                                                 'for Continuous Sign Language Recognition.')
    parser.add_argument('--datatype', default='PHONIX14',
                        type=str, help='Data type for select from "CSL or PHONIX14"')
    parser.add_argument('--data_path', default='../../data/csl-old/extract_color',
                        type=str, help='Data path for testing')
    parser.add_argument('--dict_path', default='../../data/csl-old/dictionary.txt',
                        type=str, help='Label path for testing')
    parser.add_argument('--corpus_path', default='../../data/csl-old/corpus.txt',
                        type=str, help='Label path for testing')
    parser.add_argument('--model', default='SPTransformer',
                        type=str, help='Choose a model for testing')
    parser.add_argument('--model_path', default='./models/Seq2Seq',
                        type=str, help='Model state dict path')
    parser.add_argument('--batch_size', default=2,
                        type=int, help='Batch size for testing')
    parser.add_argument('--test_batch_size', default=4,
                        type=int, help='Batch size for testing')
    parser.add_argument('--rescale', type=int, default=224,
                        help='rescale data images.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='NOTE: put num of workers to 0 to avoid memory saturation.')
    parser.add_argument('--epochs', default=30, type=int,
                        help='size of one minibatch')
    parser.add_argument('--device', type=str, default=4,
                        help='the indexes of GPUs for training or testing')
    parser.add_argument('--clip', default=1, type=int,
                        help='size of one minibatch')
    parser.add_argument('--learning_rate', default=1e-5, type=int,
                        help='size of one minibatch')
    parser.add_argument('--weight_decay', default=1e-5, type=int,
                        help='size of one minibatch')
    parser.add_argument('--log_interval', default=200, type=int,
                        help='size of one minibatch')
    parser.add_argument('--data_type', type=str, default='features',
                        help='features/resized_features/keyfeatures.')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='the interval for storing models (#epochs)')
    # Same seed for reproducibility)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    return parser
