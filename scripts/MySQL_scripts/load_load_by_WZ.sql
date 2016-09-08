USE ercot_data;
DROP TABLE IF EXISTS Load_forecast_by_WZ;
CREATE TABLE Load_forecast_by_WZ (
    delivery_date DATE,
    hour_ending TIME,
    Coast DECIMAL(9,4),
    East DECIMAL(9,4),
    FarWest DECIMAL(9,4),
    North DECIMAL(9,4),
    NorthCentral DECIMAL(9,4),
    SouthCentral DECIMAL(9,4),
    Southern DECIMAL(9,4),
    West DECIMAL(9,4),
    SystemTotal DECIMAL(9,4),
    PRIMARY KEY (delivery_date, hour_ending)
) engine = MyISAM;

LOAD DATA LOCAL INFILE '/ssd/raw_ercot_data/load_data/load_forecast_by_weather_zone/LF_by_WZ.csv'
INTO TABLE Load_forecast_by_WZ
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
(@date_variable, hour_ending, Coast, East, FarWest, North, NorthCentral, SouthCentral, Southern, West, SystemTotal) -- read one of the field to variable
SET delivery_date = STR_TO_DATE(@date_variable, '%m/%d/%Y'); -- format this date-time variable
