
# coding: utf-8


import MySQLdb
import os
import sys
import json
from MySQLdb.constants import FIELD_TYPE

my_conv = { FIELD_TYPE.DECIMAL: float }
db=MySQLdb.connect(host="localhost",db="ercot_data",read_default_file="/etc/mysql/my.cnf",conv=my_conv)
c=db.cursor()
f = open("/ssd/raw_ercot_data/dam_lmps/aggregated_LMPs/table_columns/columns.json", "r")
table_dict = json.load(f)
f.close()
f = open('create_table_queries.sql', 'w+')
print(table_dict.keys())
for table, columns in table_dict.iteritems():
    c.execute("""DROP TABLE IF EXISTS DAM_LMP%s""" % table[5:])
    c.execute("""CREATE TABLE DAM_LMP%s (
                delivery_date DATE,
                hour_ending TIME,
                PRIMARY KEY (delivery_date,hour_ending)
                ) engine = MyISAM""" % table[5:])

    for i,bus in enumerate(columns):
        if bus[0] in ['0','1','2','3','4','5','6','7','8','9'] or bus == 'LOAD':
            columns[i] = 'n' + columns[i]
        c.execute("""ALTER TABLE DAM_LMP%s ADD COLUMN %s DECIMAL(6,2)""" % (table[5:], columns[i]))


    beginning = '@date_variable,hour_ending'
    bus_vars = ''
    for bus in columns:
        bus_vars = bus_vars + ',' + '@' + bus
    column_names = beginning + bus_vars
    column_names = '(' + column_names + ')'
    null_commands = []
    for bus_var in bus_vars.split(','):
        if bus_var == '': continue
        null_commands.append(' %s = nullif(%s,\'null\')' % (bus_var[1:], bus_var))
    skeleton_query = """LOAD DATA LOCAL INFILE '/ssd/raw_ercot_data/dam_lmps/aggregated_LMPs/LMP_table%s/table.csv' INTO TABLE DAM_LMP%s FIELDS TERMINATED BY ',' LINES TERMINATED BY '\\n' %s SET delivery_date = STR_TO_DATE(@date_variable, '%%m/%%d/%%Y'),""" % (table[5:], table[5:], column_names)
    for nc in null_commands:
        skeleton_query = skeleton_query + nc + ','
    skeleton_query = skeleton_query[:-1] + ';'
    f.write(skeleton_query + '\n')
f.close()






