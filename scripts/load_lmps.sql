USE ercot_data;
DROP TABLE IF EXISTS 2016_DAM_lmps;
CREATE TABLE 2016_DAM_lmps (
    delivery_date DATE,
    hour_ending TIME,
    bus_name CHAR(128),
    lmp DECIMAL(6,2),
    PRIMARY KEY (delivery_date,hour_ending,bus_name)
) engine = MyISAM;

LOAD DATA LOCAL INFILE '/ssd/raw_ercot_data/DAM_lmps/DAM_Hr_LMP_2016/formatted_LMPs/LMPs_formatted_2016.csv' INTO TABLE 2016_DAM_lmps
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n';



