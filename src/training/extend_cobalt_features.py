__author__ = 'MarinaFomicheva'


def extend_cobalt_features(features_data, feature_data_new):

    printed = 0

    for lp in features_data.keys():

        for sys, phr in features_data[lp].keys():

            features = []

            exact_pos = 0
            coarse = 0
            diff = 0

            exact_lex = 0
            syn = 0
            para = 0
            distrib = 0

            avg_pen_ref = 0
            avg_pen_test = 0
            prop_pen = 0

            count_ref = 0
            count_test = 0
            len_ratio = 0

            for pair in features_data[lp][sys, phr]:
                attr = pair[0]
                value = float(pair[1])

                features.append(pair)

                if attr == 'count_words_ref':
                    count_ref = value

                if attr == 'count_words_test':
                    count_test = value

                if 'avg_pen' in attr and 'ref' in attr:
                    avg_pen_ref += value

                if 'avg_pen' in attr and 'test' in attr:
                    avg_pen_test += value

                if 'prop' in attr and 'pen' in attr and ('content' in attr or 'function' in attr):
                    prop_pen += value

                if 'distrib' in attr or 'syn' in attr or 'para' in attr or 'prop_aligned_exact' in attr:
                    if attr == 'prop_aligned_exact_exact':
                        exact_pos += value
                        exact_lex += value
                    elif 'coarse' in attr:
                        coarse += value
                    elif 'diff' in attr:
                        diff += value

                    if 'syn' in attr:
                        syn += value
                    elif 'para' in attr:
                        para += value
                    elif 'distrib' in attr:
                        distrib += value
                    elif '_exact_coarse' in attr:
                        exact_lex += value

            if count_ref > 0:
                len_ratio = count_test / count_ref

            features.append(tuple(['prop_aligned_pos_exact', exact_pos]))
            features.append(tuple(['prop_aligned_pos_coarse', coarse]))
            features.append(tuple(['prop_aligned_pos_diff', diff]))
            features.append(tuple(['prop_aligned_lex_exact', exact_lex]))
            features.append(tuple(['prop_aligned_lex_syn', syn]))
            features.append(tuple(['prop_aligned_lex_para', para]))
            features.append(tuple(['prop_aligned_lex_distrib', distrib]))
            features.append(tuple(['avg_pen_ref', avg_pen_ref]))
            features.append(tuple(['avg_pen_test', avg_pen_test]))
            features.append(tuple(['prop_pen', prop_pen]))
            features.append(tuple(['len_ratio', len_ratio]))

            feature_data_new[lp][sys, phr] = features

            if printed == 0:
                for feature in features:
                    print feature[0]
                    printed += 1

    return feature_data_new
