USE ercot_data;
DROP TABLE IF EXISTS DAM_SPPs;
CREATE TABLE DAM_SPPs (
    delivery_date DATE,
    hour_ending TIME,
    HB_BUSAVG DECIMAL(6,2),
    HB_HOUSTON DECIMAL(6,2),
    HB_HUBAVG DECIMAL(6,2),
    HB_NORTH DECIMAL(6,2),
    HB_SOUTH DECIMAL(6,2),
    HB_WEST DECIMAL(6,2),
    LZ_AEN DECIMAL(6,2),
    LZ_CPS DECIMAL(6,2),
    LZ_HOUSTON DECIMAL(6,2),
    LZ_LCRA DECIMAL(6,2),
    LZ_NORTH DECIMAL(6,2),
    LZ_RAYBN DECIMAL(6,2),
    LZ_SOUTH DECIMAL(6,2),
    LZ_WEST DECIMAL(6,2),
    PRIMARY KEY (delivery_date,hour_ending)
) engine = MyISAM;

LOAD DATA LOCAL INFILE '/ssd/raw_ercot_data/dam_lmps/DAM_by_LZHBSPP/spark_results/part-00000'
INTO TABLE DAM_SPPs
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
(@date_variable, hour_ending, HB_BUSAVG, HB_HOUSTON, HB_HUBAVG, HB_NORTH, HB_SOUTH, HB_WEST, LZ_AEN, LZ_CPS, LZ_HOUSTON, LZ_LCRA, LZ_NORTH, LZ_RAYBN, LZ_SOUTH, LZ_WEST) -- read one of the field to variable
SET delivery_date = STR_TO_DATE(@date_variable, '%m/%d/%Y'); -- format this date-time variable
