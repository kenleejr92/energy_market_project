__author__ = 'kenlee'

from pyspark import SparkConf, SparkContext
import json

conf = SparkConf().setMaster("local").setAppName("LMP_spark_reduction")
sc = SparkContext(conf = conf)


#Load LMP data
raw_LMP = sc.textFile('/ssd/raw_ercot_data/dam_lmps/DAM_Hr_LMP_2011/2011_test.csv')

#remove csv header
header = raw_LMP.first()
raw_LMP = raw_LMP.filter(lambda x: x!=header)

# Get all bus names
def get_bus_names(line):
    temp = line.split(",")
    return (temp[2],1)
buses = raw_LMP.map(get_bus_names)
reduced_buses = buses.reduceByKey(lambda x,y: x + y)
buses = reduced_buses.keys()
bus_list = sorted(list(buses.collect()))

#Partition the bus list into disjoint groups (1 per table)
bus_tables =[]
table_size = 1000
for i in range(0, len(bus_list), table_size):
    bus_tables.append(bus_list[i:i + table_size])
table_dict = {}
for i, table in enumerate(bus_tables):
    table_dict["table%s" % str(i)] = table
bus_table_rdd = sc.parallelize([table_dict]).map(lambda x: json.dumps(x))
bus_table_rdd.saveAsTextFile('/ssd/raw_ercot_data/dam_lmps/DAM_Hr_LMP_2011/buses')

#broadcast the list of buses to all worker nodes
broadcast_bus_tables = sc.broadcast(bus_tables)

def filter_tables(line, index):
    temp = line.split(',')
    if temp[2] in broadcast_bus_tables.value[index][:]:
        return True
    else: return False

def create_pairs(line):
        temp = line.split(",")
        return (temp[0] + "," +  temp[1], [(temp[2],temp[3])])

def check_bus_name(bus_price_list, table_index):
    bus_names = []
    for bp in bus_price_list:
        bus_names.append(bp[0])
    for bn in broadcast_bus_tables.value[table_index][:]:
        if bn not in bus_names:
            bus_price_list.append((bn, 'null'))
    return bus_price_list

def remove_kv(key_value):
        date_time = key_value[0]
        for price in key_value[1]:
             date_time = date_time + "," + price
        return date_time

for idx in range(len(bus_tables)):
    sub_rdd = raw_LMP.filter(lambda x: filter_tables(x, idx))
    #create (datetime, (bus_name,price)) pair
    pairs = sub_rdd.map(create_pairs)
    #reduce by datetime key
    reduced_by_date = pairs.reduceByKey(lambda x,y: x + y)
    # reduced_by_date.saveAsTextFile('/ssd/raw_ercot_data/dam_lmps/DAM_Hr_LMP_2011/tabler%s' % idx)
    #Fill in nulls for missing bus_names
    null_filled = reduced_by_date.mapValues(lambda x: check_bus_name(x, idx))
    #sort by bus_name for each date
    sorted_by_bus_name = null_filled.mapValues(lambda x: sorted(x, key=lambda y: y[0]))
    sorted_by_bus_name.saveAsTextFile('/ssd/raw_ercot_data/dam_lmps/DAM_Hr_LMP_2011/tableb%s' % idx)
    #delete bus_name
    no_bus_names = sorted_by_bus_name.mapValues(lambda x: [i[1] for i in x])
    #remove key-value format
    unrolled = no_bus_names.map(remove_kv)
    unrolled.saveAsTextFile('/ssd/raw_ercot_data/dam_lmps/DAM_Hr_LMP_2011/table%s' % idx)

