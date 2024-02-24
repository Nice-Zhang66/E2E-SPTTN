import os


# Return paths for image, annotation and hands(if necessary)
# For datasets Pheonix2014

def path_data(data_path=None, features_type='features', hand_query=None):
    path = os.path.join(features_type, 'fullFrame-256x256px')

    # Path for the full frame
    train_data = os.path.join(data_path, path, "train")
    valid_data = os.path.join(data_path, path, "dev")
    test_data = os.path.join(data_path, path, "test")
    hand_path = os.path.join(features_type, 'trackedRightHand-92x132px')

    # Path for hands cropped images
    if hand_query:
        train_hand = os.path.join(data_path, hand_path, "train")
        valid_hand = os.path.join(data_path, hand_path, "dev")
        test_hand = os.path.join(data_path, hand_path, "test")
    else:
        train_hand = None
        valid_hand = None
        test_hand = None

    annotations = os.path.join('annotations', 'manual')

    train_annotations = os.path.join(data_path, annotations, "train.corpus.csv")
    valid_annotations = os.path.join(data_path, annotations, "dev.corpus.csv")
    test_annotations = os.path.join(data_path, annotations, "test.corpus.csv")

    # Returns the full frame path + annotations of (train, dev, test)
    return (train_data, train_annotations, train_hand), (valid_data, valid_annotations, valid_hand), (
            test_data, test_annotations, test_hand)
