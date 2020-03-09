#!/usr/bin/python3

from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
import sys

class Searcher():
    def __init__(self):
        self.conf = SparkConf().setMaster("local").setAppName("Searcher")
        self.sc = SparkContext(conf = self.conf)

    def load_data(self, data_file):
        raw_data = self.sc.textFile(data_file)
        fields = raw_data.map(lambda x: x.split("\t"))
        self.documents = fields.map(lambda x: x[3].split(" "))
        self.document_names = fields.map(lambda x: x[1])

    def hashing(self, size):
        self.hashing_TF = HashingTF(size)  #100K hash buckets just to save some memory
        tf = self.hashing_TF.transform(self.documents)

        tf.cache()
        idf = IDF(minDocFreq=2).fit(tf)
        self.tfidf = idf.transform(tf)

    def search(self, search_text):
        search_text_TF = self.hashing_TF.transform([search_text])
        search_text_hash_value = int(search_text_TF.indices[0])
        search_text_relevance = self.tfidf.map(lambda x: x[search_text_hash_value])

        return search_text_relevance.zip(self.document_names)

def main(argv):
    searcher = Searcher()
    searcher.load_data(argv[0])
    searcher.hashing(100000)
    zipped_results = searcher.search(argv[1])
    print("Best document for {} is:".format(argv[1]))
    print(zipped_results.max())
    for row in zipped_results.collect():
        print(row)

if __name__ == '__main__':
    main(sys.argv[1:])