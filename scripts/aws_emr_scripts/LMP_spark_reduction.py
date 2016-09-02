from pyspark import SparkConf, SparkContext


conf = SparkConf().setMaster("local").setAppName("LMP_spark_reduction")
sc = SparkContext(conf = conf)


#Load LMP data
raw_LMP = sc.textFile("/ssd/raw_ercot_data/dam_lmps/DAM_by_LZHBSPP/2010-2016_DAM_by_SPP.csv")

#remove csv header
header = raw_LMP.first()
raw_LMP = raw_LMP.filter(lambda x: x!=header)

# # Get all bus names
# def get_bus_names(line):
#     temp = line.split(",")
#     return (temp[2],1)
# buses = raw_LMP.map(get_bus_names)
# buses = buses.reduceByKey(lambda x,y: x + y).keys()
# bus_list = list(buses.collect())
# #broadcast the list of buses to all worker nodes
# bus_list = sc.broadcast(bus_list)

#create (datetime, (bus_name,price)) pair
def create_pairs(line):
    temp = line.split(",")
    return (temp[0] + "," +  temp[1], [(temp[3],temp[4])])
pairs = raw_LMP.map(create_pairs)

#reduce by datetime key
reduced_by_date = pairs.reduceByKey(lambda x,y: x + y)

#sort by bus_name for each date
sorted_by_bus_name = reduced_by_date.mapValues(lambda x: sorted(x, key=lambda y: y[0]))

#get bus_names in order
first_date = sorted_by_bus_name.take(1)
bus_names = []
for bus_price in first_date[0][1]:
    bus_names.append(bus_price[0])
bus_names_rdd = sc.parallelize(bus_names).coalesce(1)
bus_names_rdd.saveAsTextFile("/ssd/raw_ercot_data/dam_lmps/DAM_by_LZHBSPP/zones")

#delete bus_name
no_bus_names = sorted_by_bus_name.mapValues(lambda x: [i[1] for i in x])

#remove key-value format
def remove_kv(key_value):
    date_time = key_value[0]
    for price in key_value[1]:
         date_time = date_time + "," + price
    return date_time
unrolled = no_bus_names.map(remove_kv) 
unrolled.saveAsTextFile("/ssd/raw_ercot_data/dam_lmps/DAM_by_LZHBSPP/spark_results")











