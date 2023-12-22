def load_dataset(root, dataset, split_method):
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
    test_txt = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, 'r') as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, 'r') as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids, val_img_ids, test_txt


def load_param(channel_size):
    if channel_size == 'one':
        num_filters = [4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 128]
    elif channel_size == 'two':
        num_filters = [8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 256]
    elif channel_size == 'three':
        num_filters = [16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 512]
    elif channel_size == 'four':
        num_filters = [32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024]

    return num_filters
