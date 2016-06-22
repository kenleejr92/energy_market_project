__author__ = 'kenlee'
import MySQLdb

db=MySQLdb.connect(host="localhost",db="ercot_data",read_default_file="/etc/mysql/my.cnf")
db.query("""SELECT spp FROM DAM_prices_by_SPP
         WHERE settlement_point = "LZ_CPS"
         AND delivery_date > "2015-01-01"
         AND delivery_date < "2015-12-31"
         ORDER BY delivery_date,hour_ending""")

r=db.store_result()
rows = r.fetch_row(maxrows=0)
print type(rows)