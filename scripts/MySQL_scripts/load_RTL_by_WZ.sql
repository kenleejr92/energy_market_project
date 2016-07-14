USE ercot_data;
DROP TABLE IF EXISTS RTL_by_LZ;
CREATE TABLE RTL_by_LZ (
    delivery_date DATE,
    hour_ending TIME,
    LZ_WEST_RTL DECIMAL(9,4),
    LZ_NORTH_RTL DECIMAL(9,4),
    LZ_SOUTH_RTL DECIMAL(9,4),
    LZ_HOUSTON_RTL DECIMAL(9,4),
    PRIMARY KEY (delivery_date, hour_ending)
) engine = MyISAM;

LOAD DATA LOCAL INFILE '/ssd/raw_ercot_data/load_data/real_time_demand/RTL_by_LZ/part-00000'
INTO TABLE RTL_by_LZ
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
(@date_variable, hour_ending, LZ_WEST_RTL, LZ_NORTH_RTL, LZ_SOUTH_RTL, LZ_HOUSTON_RTL) -- read one of the field to variable
SET delivery_date = STR_TO_DATE(@date_variable, '%m/%d/%Y'); -- format this date-time variable
