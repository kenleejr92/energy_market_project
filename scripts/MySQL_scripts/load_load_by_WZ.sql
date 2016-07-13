USE ercot_data;
DROP TABLE IF EXISTS Load_by_LZ;
CREATE TABLE Load_by_LZ (
    delivery_date DATE,
    hour_ending TIME,
    LZ_WEST_load DECIMAL(9,4),
    LZ_NORTH_load DECIMAL(9,4),
    LZ_SOUTH_load DECIMAL(9,4),
    LZ_HOUSTON_load DECIMAL(9,4),
    PRIMARY KEY (delivery_date, hour_ending)
) engine = MyISAM;

LOAD DATA LOCAL INFILE '/ssd/raw_ercot_data/load_data/load_forecast_by_weather_zone/LF_by_LZ/LF_by_LZ.csv'
INTO TABLE Load_by_LZ
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
(@date_variable, hour_ending, LZ_WEST_load, LZ_NORTH_load, LZ_SOUTH_load, LZ_HOUSTON_load) -- read one of the field to variable
SET delivery_date = STR_TO_DATE(@date_variable, '%m/%d/%Y'); -- format this date-time variable
