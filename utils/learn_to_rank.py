__author__ = 'MarinaFomicheva'


class LearnToRank(object):

    @staticmethod
    def training_set(work_dir, data, features, ranks, name):

        with open(work_dir + '/' + 'x_' + name + '.tsv', 'w') as x_file:
            with open(work_dir + '/' + 'y_' + name + '.tsv', 'w') as y_file:

                for dataset_name, lang_pair in sorted(ranks.keys()):
                    for human_comparison in ranks[dataset_name, lang_pair]:

                        if human_comparison.sign == '=':
                            continue

                        seg_id = human_comparison.phrase
                        winner, loser = human_comparison.winner_loser()
                        idx_winner = data.get_sentence_idx(dataset_name, lang_pair, seg_id, winner)
                        idx_loser = data.get_sentence_idx(dataset_name, lang_pair, seg_id, loser)

                        positive_instance, negative_instance = self.get_instance(feature_values[idx_winner],
                                                                         feature_values[idx_loser])

                        f_features.write('\t'.join([str(x) for x in positive_instance]) + '\n')
                        f_features.write('\t'.join([str(x) for x in negative_instance]) + '\n')

                        f_objective.write('1' + '\n')
                        f_objective.write('0' + '\n')

                f_features.close()
                f_objective.close()

