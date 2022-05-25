
import csv
import os

class_counters = {}

with open('commands.csv','rt')as f:
  data = csv.reader(f)
  next(data, None)  # skip the headers
  for row in data:
        cls = row[1]
        if not cls in class_counters:
          class_counters[cls] = 1
        else:
          class_counters[cls] += 1

        if not os.path.exists(cls):
          os.makedirs(cls)

        with open(os.path.join(str(cls), str(class_counters[cls]) + ".txt" ),'w') as s:
          s.write(row[0])
          s.close()
