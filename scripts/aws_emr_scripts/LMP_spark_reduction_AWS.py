from pyspark import SparkConf, SparkContext

sc = SparkContext(appName="LMP_spark_reduction")

#Load LMP data
raw_LMP = sc.textFile("s3://lmpdata/2011_LMPs/2011_LMPs.csv")

#remove csv header
header = raw_LMP.first()
raw_LMP = raw_LMP.filter(lambda x: x!=header)

#create (datetime, (bus_name,price)) pair
def create_pairs(line):
    temp = line.split(",")
    return (temp[0] + " " +  temp[1], [(temp[2],temp[3])])
pairs = raw_LMP.map(create_pairs)

#reduce by datetime key
#pairs.partitionBy(2190,rangePartitioner)
reduced_by_date = pairs.reduceByKey(lambda x,y: x + y)

#sort by bus_name for each date
sorted_by_bus_name = reduced_by_date.mapValues(lambda x: sorted(x, key=lambda y: y[0]))

#get and save bus_names in order for MySQL table creation later
first_date = sorted_by_bus_name.take(1)
bus_names = []
for bus_price in first_date[0][1]:
    bus_names.append(bus_price[0])
bus_names_rdd = sc.parallelize(bus_names).coalesce(1)
bus_names_rdd.saveAsTextFile("s3://lmpdata/2011_LMPs/bus_names")

#delete bus_name
no_bus_names = sorted_by_bus_name.mapValues(lambda x: [i[1] for i in x])

#remove key-value format
def remove_kv(key_value):
    date_time = key_value[0]
    for price in key_value[1]:
         date_time = date_time + "," + price
    return date_time
unrolled = no_bus_names.map(remove_kv) 
unrolled.saveAsTextFile("s3://lmpdata/2011_LMPs/formatted_LMPs");











