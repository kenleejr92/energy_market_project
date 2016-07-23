USE ercot_data;
DROP TABLE IF EXISTS crr_ownership;
CREATE TABLE crr_ownership (
    CRR_ID INT,
    AccountHolder CHAR(15),
    Category CHAR(15),
    HedgeType CHAR(15),
    CRRType CHAR(15),
    Source CHAR(15),
    Sink CHAR(15),
    StartDate DATE,
    EndDate DATE,
    TimeOfUse CHAR(15),
    MW DECIMAL(5,1),
    PRIMARY KEY (CRR_ID)
) engine = MyISAM;

LOAD DATA LOCAL INFILE '/ssd/raw_ercot_data/crrs/CRR_ownership_or_record/crr_ownership_of_record.csv' INTO TABLE crr_ownership
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
(CRR_ID, @dummy1, AccountHolder, Category, HedgeType, CRRType, Source, Sink, @dummy2, @start, @end, TimeOfUse, MW)
SET StartDate = STR_TO_DATE(@start, '%m/%d/%Y'), EndDate = STR_TO_DATE(@end, '%m/%d/%Y');



