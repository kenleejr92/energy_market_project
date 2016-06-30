import MySQLdb
from MySQLdb.constants import FIELD_TYPE

class Query_ERCOT_DB(object):
    #initialization of MySQLdb, shared by all instances of Query_ERCOT_DB
    my_conv = { FIELD_TYPE.DECIMAL: float }
    db=MySQLdb.connect(host="localhost",db="ercot_data",read_default_file="/etc/mysql/my.cnf",conv=my_conv)
    c=db.cursor()
        
    