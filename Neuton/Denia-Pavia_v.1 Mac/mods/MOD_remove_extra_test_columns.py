def remove_extra_test_columns(data):
    '''
    Removes test columns that are not present in train set
    If test set does not include one or more columns that are present in train set
    an error is printed and code execution interrupts
    '''
    import pickle
    import numpy as np
    import sys
    import timeit
    start = timeit.default_timer()
    
    pickle_in = open('dict/original_train_header.p','rb')
    original_train_header = pickle.load(pickle_in)
    cols_to_drop = []
    for i, r in enumerate(data.columns.isin(list(original_train_header))):
        if r == False:
            cols_to_drop.append(i)
    data.drop(data.columns[cols_to_drop], axis = 1, inplace = True)
    try:
        np.all(list(original_train_header) == data.columns)
        stop = timeit.default_timer()
        print('\n   - Drop columns not present in Train set finished, time: {:.2f} minutes'.format((stop - start) / 60))
        print('       Number of columns droped: ', len(cols_to_drop))
        print('    ','-'*55)
    except:
        print('\nError: Test set excludes one or more Train set columns\n')
        sys.exit(1)       
    sys.stdout.flush()
    return data


