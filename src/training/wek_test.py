__author__ = 'MarinaFomicheva'

from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
import weka.core.jvm as jvm
import javabridge
jvm.start()


class Logistic(Classifier):
    """
    Wrapper class for classifiers that use a single base classifier.
    """

    def __init__(self, jobject=None, options=None):
        """
        Initializes the specified classifier using either the classname or the supplied JB_Object.

        :param classname: the classname of the classifier
        :type classname: str
        :param jobject: the JB_Object to use
        :type jobject: JB_Object
        :param options: the list of commandline options to set
        :type options: list
        """
        classname = "weka.classifiers.functions.Logistic"

        if jobject is None:
            jobject = Classifier.new_instance(classname)
        self.enforce_type(jobject, "weka.classifiers.functions.Logistic")
        super(Logistic, self).__init__(classname=classname, jobject=jobject, options=options)



    @property
    # def coefficients(self):
    #     m = javabridge.call(self.jobject, "coefficients", "()[[D")
    #     objs = javabridge.get_env().get_object_array_elements(m)
    #     coeffs = []
    #     for obj in objs:
    #         coeffs.append(javabridge.get_env().get_double_array_elements(obj))
    #
    #     return coeffs
    def coefficients(self):
        m = javabridge.call(self.jobject, "coefficients", "()[[D")
        result = [javabridge.get_env().get_array_length(m)]
        rows = javabridge.get_env().get_object_array_elements(m)

        for row in rows:
            elements = []
            for i, element in enumerate(javabridge.get_env().get_double_array_elements(row)):
                elements.append(float(element))
            result.append(elements)

        return result

tmp_file = open('tmp.txt', 'w')
loader = Loader(classname = "weka.core.converters.ArffLoader")
data = loader.load_file("/Users/MarinaFomicheva/Dropbox/workspace/training/cobalt_features_arff/training_set.newstest2014.cs-en.arff")
data.class_is_last()
cls = Logistic()
cls.build_classifier(data)
print >>tmp_file, cls

eval = Evaluation(data)
eval.test_model(cls, data)



for idx, inst in enumerate(data):

    pred = cls.classify_instance(inst)
    dist = cls.distribution_for_instance(inst)
    print str(idx + 1) + ' ' + str(pred) + ' ' + str(dist)










