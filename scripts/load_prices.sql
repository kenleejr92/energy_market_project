USE ercot_data;
DROP TABLE IF EXISTS 2010_prices;
CREATE TABLE 2010_prices (
    delivery_date DATE,
    hour_ending TIME,
    bus_name CHAR(128),
    lmp DECIMAL(5,2),
    PRIMARY KEY (delivery_date,hour_ending,bus_name)
) engine=innoDB;

LOAD DATA LOCAL INFILE '/home/kenlee/Energy_Market_Project/LMP_data/DAM_Hr_LMP_2010/formatted_spark/part-00000' INTO TABLE 2010_prices 
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n';

LOAD DATA LOCAL INFILE '/home/kenlee/Energy_Market_Project/LMP_data/DAM_Hr_LMP_2010/formatted_spark/part-00001' INTO TABLE 2010_prices
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n';

LOAD DATA LOCAL INFILE '/home/kenlee/Energy_Market_Project/LMP_data/DAM_Hr_LMP_2010/formatted_spark/part-00002' INTO TABLE 2010_prices
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n';

LOAD DATA LOCAL INFILE '/home/kenlee/Energy_Market_Project/LMP_data/DAM_Hr_LMP_2010/formatted_spark/part-00003' INTO TABLE 2010_prices
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n';

