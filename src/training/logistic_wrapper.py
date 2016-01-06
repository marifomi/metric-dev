__author__ = 'MarinaFomicheva'

from weka.classifiers import Classifier
import javabridge

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

    def save_model(self, output_directory, dataset, lang_pair, feature_set):
        file = open(output_directory + '/' + feature_set + '.' + dataset + '.' + lang_pair + '.out', 'w')
        print >>file, self

