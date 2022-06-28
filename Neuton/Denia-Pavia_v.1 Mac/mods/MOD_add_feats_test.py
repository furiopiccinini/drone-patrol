import pickle, timeit, sys, os

def add_feats_test(data):
    from MOD_feats_divide_lineary_coefficients_test import feats_divide_lineary_coefficients_test
    from MOD_combine_xgb_feats_test import combine_xgb_feats_test
    from MOD_text_preprocessing import dump

    start = timeit.default_timer()

    # check if FE was not performed on train due to only 1 available independent variable in train
    with open('dict/perform_feature_engineering.p','rb') as pickle_in:
        perform_feature_engineering = pickle.load(pickle_in)

    if perform_feature_engineering:
        try:
            with open('dict/mutual_corr_cols.p','rb') as pickle_in:
                mutual_corr_cols = pickle.load(pickle_in)
        except:
            pass
        path = 'dict/methods_final.p'
        if not os.path.exists(path):
#            dump(data, 'output/processed_test.csv')
            return  data


        with open(path,'rb') as pickle_in:
            methods = pickle.load(pickle_in)
        test2 = data.copy()    
        if len(methods) > 0:
            print('\n')
            print('='*60)
            print('Feature engineering on Test set started')
            for i in range(len(methods)):
                if methods[i] == 'combine_xgb_feats':
                    test2 = locals()['combine_xgb_feats_test'](test2)
                elif methods[i] == 'feats_divide_lineary_coefficients':
                    test2 = locals()['feats_divide_lineary_coefficients_test'](test2)
                else: 
                    test2 = test2.drop(set(mutual_corr_cols), axis = 1)
                    print('\n   - Exclude_mutual_cors completed')
                    print('    ','-'*55)
#                dump(test2, 'output/processed_test.csv')
                #test2.to_csv('output/processed_test.csv', index = False)
            print('\n   - Feature engineering yielded better results\n       processed_test.csv is saved to output folder')
            stop = timeit.default_timer()
            print('\nFeature engineering on Test set finished, time: {:.2f} minutes'.format((stop - start) / 60))
            print('='*60)
        else:
#            dump(test2, 'output/processed_test.csv')
            #test2.to_csv('output/processed_test.csv', index = False)
            print('\n   - Data preprocessing baseline metric was not improved during feature engineering on Train set\n       processed_test.csv is saved to output folder')        
        sys.stdout.flush()
        return test2
    else:
        print('    - No features were added due to only one column in train and (probably) test data')
 #       dump(data, 'output/processed_test.csv')
        os.system('say "test feature engineering has finished"')
        return data

