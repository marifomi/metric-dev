import xml.etree.ElementTree as ET
import os


class FeaturesReader(object):

    @staticmethod
    def read_features(file_):

        tree = ET.parse(file_)
        root = tree.getroot()

        features = []

        for f in root.findall('feature'):
            features.append(f.get('index'))

        return features


def main():

    my_file = os.path.expanduser('~/Dropbox/workspace/questplusplus/config/features/my_features_fluency_sentence_level.xml')
    reader = FeaturesReader()
    features = reader.read_features(my_file)
    for f in features:
        print(f)


if __name__ == '__main__':
    main()