USE ercot_data;
DROP TABLE IF EXISTS natural_gas_prices;
CREATE TABLE natural_gas_prices (
    delivery_date DATE,
    price DECIMAL(6,2), # Dollars per thousand cubic feet
    PRIMARY KEY (delivery_date)
) engine = MyISAM;

LOAD DATA LOCAL INFILE '/ssd/raw_ercot_data/natural_gas/TX_NG_prices.csv' INTO TABLE natural_gas_prices
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n';



