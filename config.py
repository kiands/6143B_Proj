pretrained_model = 'models'  # pretrained model
checkpoint = 'output/'  # the path to save the model
train_set = './datasets/train.csv'  # the path of train
seed = 42  # random_seed
proportion = 0.8  # train:valid = 4:1
epochs = 50
batch_size = 16
labels_counts = 2  # classes
max_length = 128  # max sentence length
bert_lr = 0.00002 # bert learning rate
cls_lr = 0.001 # classifier learning rate
decay_factor = 0.01 # parameter for weight_decay