from pyspark import SparkConf, SparkContext
from datetime import datetime



conf = SparkConf().setMaster("local").setAppName("LMP_spark_reduction")
sc = SparkContext(conf = conf)

def format(line):
    temp = line.split(",")
    date = temp[0].split("/");
    new_date = date[2] + "/" + date[1] + "/" + date[0];    
    return [new_date + "," + temp[1] + "," + temp[2] + "," + temp[3]];


raw_LMP = sc.textFile("/home/kenlee/Energy_Market_Project/LMP_data/DAM_Hr_LMP_2010/merge_test.csv")
#remove csv header
header = raw_LMP.first()
raw_LMP = raw_LMP.filter(lambda x: x!=header)
formatted_LMP = raw_LMP.flatMap(format)
formatted_LMP.saveAsTextFile("/home/kenlee/Energy_Market_Project/LMP_data/DAM_Hr_LMP_2010/formatted_merge_test");











