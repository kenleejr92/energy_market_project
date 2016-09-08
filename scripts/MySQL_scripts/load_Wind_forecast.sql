USE ercot_data;
DROP TABLE IF EXISTS Wind_forecasts;
CREATE TABLE Wind_forecasts (
    delivery_date DATETIME,
    ACTUAL_HOUSTON DECIMAL(7,2),
    STWPF_HOUSTON DECIMAL(7,2),
    WGRPP_HOUSTON DECIMAL(7,2),
    ACTUAL_NORTH DECIMAL(7,2),
    STWPF_NORTH DECIMAL(7,2),
    WGRPP_NORTH DECIMAL(7,2),
    ACTUAL_SOUTH DECIMAL(7,2),
    STWPF_SOUTH DECIMAL(7,2),
    WGRPP_SOUTH DECIMAL(7,2),
    ACTUAL_WEST DECIMAL(7,2),
    STWPF_WEST DECIMAL(7,2),
    WGRPP_WEST DECIMAL(7,2),
    PRIMARY KEY (delivery_date)
) engine = MyISAM;

LOAD DATA LOCAL INFILE '/ssd/raw_ercot_data/wind_data/Wind_forecasts.csv'
INTO TABLE Wind_forecasts
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
(@date_variable, ACTUAL_HOUSTON, STWPF_HOUSTON, WGRPP_HOUSTON, ACTUAL_NORTH, STWPF_NORTH, WGRPP_NORTH, ACTUAL_SOUTH, STWPF_SOUTH, WGRPP_SOUTH,
    ACTUAL_WEST,
    STWPF_WEST,
    WGRPP_WEST, @dummy) -- read one of the field to variable
SET delivery_date = STR_TO_DATE(@date_variable, '%m/%d/%Y %H:%i'); -- format this date-time variable
