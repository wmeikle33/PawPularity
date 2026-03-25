def stratified_split(data, n_splits, test_size):
    sssplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for train_index, test_index in sssplit.split(data, data['Pawpularity']):
        training_set = data.iloc[train_index]
        eval_set = data.iloc[test_index]
