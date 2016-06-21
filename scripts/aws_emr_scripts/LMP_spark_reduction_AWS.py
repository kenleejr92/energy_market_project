from pyspark import SparkContext

sc = SparkContext(appName="LMP_spark_reduction_AWS")
def format(line):
    temp = line.split(",")
    date = temp[0].split("/")
    new_date = date[2] + "/" + date[0] + "/" + date[1]
    return [new_date + "," + temp[1] + "," + temp[2] + "," + temp[3]]
raw_LMP = sc.textFile("s3://lmpdata/2010_LMPs/2010_LMPs.csv")
#remove csv header
header = raw_LMP.first()
raw_LMP = raw_LMP.filter(lambda x: x!=header)
formatted_LMP = raw_LMP.flatMap(format)
formatted_LMP.saveAsTextFile("s3://lmpdata/2010_LMPs/formatted_2010_LMPs")
sc.stop()
