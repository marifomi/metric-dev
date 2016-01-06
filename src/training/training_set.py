__author__ = 'MarinaFomicheva'

class TrainingSet(object):

    def __init__(self):
        self.training_instances = []

    def cross_human_metric_absolute(self, human_data, features_data, direction):

        for system, phrase in sorted(features_data[direction].keys(), key=lambda element: (element[0], element[1])):

            if not (system, phrase) in human_data[direction].keys():
                continue

            self.training_instances.append([x[1] for x in features_data[direction][system, phrase]] + human_data[direction][system, phrase])

    def cross_human_metric(self, human_data, features_data, directions):

        for direction in directions:

            for case in human_data[direction]:
                if case.sign == '=':
                    continue

                good_system = features_data[direction][(self.find_winner(case), case.phrase)]
                bad_system = features_data[direction][(self.find_loser(case), case.phrase)]
                self.training_instances.append(self.get_positive_instance(good_system, bad_system))
                self.training_instances.append(self.get_negative_instance(good_system, bad_system))

    def build_from_file(self, features_data, human_data, dataset, direction, output_directory, max_segments):

        output_file_name = 'training_set.' + dataset + '.' + direction + '.arff'
        self.cross_human_metric(human_data, features_data, [direction])
        self.make_arff(self.get_feature_names(features_data), dataset, output_directory, output_file_name)

    def build_absolute(self, features_data, human_data, directions, output_file):

        for direction in directions:
            self.cross_human_metric_absolute(human_data, features_data, direction)
            for list in self.training_instances:
                print >>output_file, '\t'.join([str(x) for x in list])
            output_file.close()

    def merge_from_files(self, features_data, human_data, dataset, directions, output_directory, feature_set):

        file_name = feature_set + '.' + dataset + '.' + 'all_lp.arff'
        self.cross_human_metric(human_data, features_data, directions)
        self.make_arff(self.get_feature_names(features_data), dataset, output_directory, file_name)

    def find_winner(self, case):

        if case.sign == '<':
            return case.sys1
        else:
            return case.sys2

    def find_loser(self, case):

        if case.sign == '<':
            return case.sys2
        else:
            return case.sys1

    def get_positive_instance(self, good, bad):

        new_feature_vector = []

        for i, feature in enumerate(good):
            positive_value = float(feature[1])
            negative_value = float(bad[i][1])
            new_feature_vector.append(positive_value - negative_value)

        new_feature_vector.append('positive')
        return new_feature_vector

    def get_negative_instance(self, good, bad):

        new_feature_vector = []

        for i, feature in enumerate(good):
            positive_value = float(feature[1])
            negative_value = float(bad[i][1])
            new_feature_vector.append(negative_value - positive_value)

        new_feature_vector.append('negative')
        return new_feature_vector

    def make_arff(self, feature_names, dataset, output_directory, file_name):

        file = open(output_directory + '/' + file_name, 'w')

        print >>file, '@relation ' + dataset

        for name in feature_names:
            print >>file, '@attribute ' + name + ' real'

        print >>file, '@attribute ' + 'class' + ' {positive, negative}'
        print >>file, '@data'

        for instance in self.training_instances:
            print >>file, ','.join([str(x) for x in instance])

        file.close()

    def get_feature_names(self, features_data):

        names = []

        for lang in features_data.keys():
            for sys, phrase in features_data[lang].keys():
                for feature in features_data[lang][sys, phrase]:
                    names.append(feature[0])
                break

        return names
