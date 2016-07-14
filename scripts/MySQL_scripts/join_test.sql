SELECT * FROM DAM_SPPs
INNER JOIN Load_by_LZ 
USING (delivery_date,hour_ending)
INNER JOIN RTL_by_LZ
USING (delivery_date,hour_ending)
WHERE DAM_SPPs.delivery_date > "2012-01-01"
AND DAM_SPPs.delivery_date < "2012-12-31"
ORDER BY DAM_SPPs.delivery_date, DAM_SPPs.hour_ending

