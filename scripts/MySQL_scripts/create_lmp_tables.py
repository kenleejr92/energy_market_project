
# coding: utf-8

# In[20]:

import MySQLdb
import os
import sys
from MySQLdb.constants import FIELD_TYPE

my_conv = { FIELD_TYPE.DECIMAL: float }
db=MySQLdb.connect(host="localhost",db="ercot_data",read_default_file="/etc/mysql/my.cnf",conv=my_conv)
c=db.cursor()


# In[21]:

c.execute("""DROP TABLE IF EXISTS DAM_SPPs""")
c.execute("""CREATE TABLE DAM_SPPs (
            delivery_date DATE,
            hour_ending TIME,
            PRIMARY KEY (delivery_date,hour_ending)
            ) engine = MyISAM""")


# In[22]:

f = open("/ssd/raw_ercot_data/dam_lmps/DAM_by_LZHBSPP/zones/part-00000","r")
bus_names = [line[:-1] for line in f]
f.close()
bus_names


# In[23]:

for bus in bus_names:
    c.execute("""ALTER TABLE DAM_SPPs ADD COLUMN `%s_SPP` DECIMAL(6,2)""" % bus)


# In[25]:

c.execute("""LOAD DATA LOCAL INFILE '/ssd/raw_ercot_data/dam_lmps/DAM_by_LZHBSPP/spark_results/part-00000' 
INTO TABLE DAM_SPPs FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n' 
(@date_variable,hour_ending,HB_BUSAVG_SPP,HB_HOUSTON_SPP,HB_HUBAVG_SPP,HB_NORTH_SPP,HB_SOUTH_SPP,HB_WEST_SPP,LZ_AEN_SPP,LZ_CPS_SPP,LZ_HOUSTON_SPP,LZ_LCRA_SPP,LZ_NORTH_SPP,LZ_RAYBN_SPP,LZ_SOUTH_SPP,LZ_WEST_SPP)
SET delivery_date = STR_TO_DATE(@date_variable, '%m/%d/%Y');
""")


# In[ ]:



