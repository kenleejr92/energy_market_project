SELECT * FROM (
    (SELECT delivery_date,hour_ending,spp FROM DAM_prices_by_SPP 
    WHERE settlement_point = "LZ_CPS" 
    AND delivery_date > "2015-01-01" 
    AND delivery_date < "2015-12-31" 
    ORDER BY delivery_date,hour_ending) AS t1
    INNER JOIN 
    (SELECT delivery_date,hour_ending,spp FROM DAM_prices_by_SPP
    WHERE settlement_point = "LZ_HOUSTON" 
    AND delivery_date > "2015-01-01" 
    AND delivery_date < "2015-12-31" 
    ORDER BY delivery_date,hour_ending) AS t2
    USING(delivery_date,hour_ending)
    ); 

