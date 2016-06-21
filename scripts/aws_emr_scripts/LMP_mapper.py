#!/usr/bin/env python

import sys

for line in sys.stdin:
    line = line.split(",")
    if line[0] == "DeliveryDate":
        continue
    else:
        date = line[0].split("/")
        new_date = date[2] + "/" + date[0] + "/" + date[1]
        print new_date + "," + line[1] + "," + line[2] + "," + line[3]

