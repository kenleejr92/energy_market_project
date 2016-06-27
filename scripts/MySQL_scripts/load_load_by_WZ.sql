USE ercot_data;
DROP TABLE IF EXISTS DAM_prices_by_SPP;
CREATE TABLE DAM_prices_by_SPP (
    delivery_date DATE,
    hour_ending TIME,
    settlement_point CHAR(128),
    coast DECIMAL(6,2),
    east DECIMAL(,
    farwest,
    north,
    north_central,
    south_central,
    southern,
    west,
    system_total,
    PRIMARY KEY (delivery_date,hour_ending)
) engine = MyISAM;

LOAD DATA LOCAL INFILE '/ssd/raw_ercot_data/DAM_lmps/DAM_by_LZHBSPP/2010-2016_DAM_by_SPP.csv'
INTO TABLE DAM_prices_by_SPP
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
(@date_variable, hour_ending, @dummy, settlement_point, spp) -- read one of the field to variable
SET delivery_date = STR_TO_DATE(@date_variable, '%m/%d/%Y'); -- format this date-time variable
