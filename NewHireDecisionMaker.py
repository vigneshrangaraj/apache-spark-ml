#!/usr/bin/python3

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from numpy import array
import sys

def binary(YN):
    if (YN == 'Y'):
        return 1
    else:
        return 0

def mapEducation(degree):
    if (degree == 'BS'):
        return 1
    elif (degree =='MS'):
        return 2
    elif (degree == 'PhD'):
        return 3
    else:
        return 0

# Convert a list of raw fields from our CSV file to a
# LabeledPoint that MLLib can use. All data must be numerical...
def createLabeledPoints(fields):
    yearsExperience = int(fields[0])
    employed = binary(fields[1])
    previousEmployers = int(fields[2])
    educationLevel = mapEducation(fields[3])
    topTier = binary(fields[4])
    interned = binary(fields[5])
    hired = binary(fields[6])

    return LabeledPoint(hired, array([yearsExperience, employed,
        previousEmployers, educationLevel, topTier, interned]))

class NewHireDecisionMaker():
    def __init__(self):
        self.conf = SparkConf().setMaster("local").setAppName("NewHireDecisionMaker")
        self.sc = SparkContext(conf = self.conf)

    def load_traing_data(self, training_data_file):
        raw_data = self.sc.textFile(training_data_file)
        header = raw_data.first()
        raw_data = raw_data.filter(lambda x:x != header)
        csv_data = raw_data.map(lambda x: x.split(","))

        # Convert these lists to LabeledPoints
        self.training_data = csv_data.map(createLabeledPoints)

    def make_model(self):
        self.model = DecisionTree.trainClassifier(self.training_data, numClasses=2,
                                     categoricalFeaturesInfo={1:2, 3:4, 4:2, 5:2},
                                     impurity='gini', maxDepth=5, maxBins=32)

    def make_predictions(self, test_data):
        transformed_test_data = self.sc.parallelize(test_data)
        return self.model.predict(transformed_test_data)
        
    def show_model(self):
        return self.model.toDebugString()

def main(argv):
    new_hire = NewHireDecisionMaker()
    new_hire.load_traing_data(argv[0])
    new_hire.make_model()

    test_data = [ array([10, 1, 3, 1, 0, 0])]
    predictions = new_hire.make_predictions(test_data)
    
    print('Hire prediction:')

    for result in predictions.collect():
        print(">>>: " + str(result))

    print("\n*** Model ***")
    print(new_hire.show_model())

if __name__ == '__main__':
    main(sys.argv[1:])