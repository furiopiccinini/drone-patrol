import pickle


def ts_test_mean_target(data):
    pickle_in = open(
        'dict/dict_for_ts_test_mean_target.p', 'rb')
    mean_target_dict = pickle.load(pickle_in)

    for i in mean_target_dict.keys():
        data[i] = data[i].map(mean_target_dict[i])
    print('\n   - Test time seris mean target encoding completed')
    print('-' * 50)
    return data
