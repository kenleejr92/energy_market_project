__author__ = 'kenlee'

from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster('local').setAppName('Load_spark_local')
sc = SparkContext(conf = conf)

loads_by_WZ = sc.textFile('/ssd/raw_ercot_data/load_data/real_time_demand/RTL_by_WZ/RTL_by_WZ.csv')
#remove csv header
header = loads_by_WZ.first()
header2 = 'OperDay,HourEnding,COAST,EAST,FAR_WEST,NORTH,NORTH_C,SOUTHERN,SOUTH_C,WEST,TOTAL,DSTFlag'
loads_by_WZ = loads_by_WZ.filter(lambda x: x!=header and x!=header2)

#filter out zero values
def filter_out_zeroes(line):
    fields = line.split(',')
    if(float(fields[2]) != 0 and
       float(fields[3]) != 0 and
       float(fields[4]) != 0 and
       float(fields[5]) != 0 and
       float(fields[6]) != 0 and
       float(fields[7]) != 0 and
       float(fields[8]) != 0 and
       float(fields[9]) != 0):
        return line

loads_by_WZ = loads_by_WZ.filter(filter_out_zeroes)


def convert_to_LZ(line):
    fields = line.split(',')
    delivery_date = fields[0]
    hour_ending = fields[1]
    LZ_west = float(fields[4]) + float(fields[9])
    LZ_north = float(fields[3]) + float(fields[5]) + float(fields[6])
    LZ_south = float(fields[7]) + float(fields[8])
    LZ_houston = float(fields[2])
    return delivery_date + ',' + hour_ending + ',' + str(LZ_west) + ',' + str(LZ_north) + ',' + str(LZ_south) + ',' + str(LZ_houston)

loads_by_LZ = loads_by_WZ.map(convert_to_LZ)
loads_by_LZ.saveAsTextFile('/ssd/raw_ercot_data/load_data/real_time_demand/RTL_by_LZ')