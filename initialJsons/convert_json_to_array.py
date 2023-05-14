import re
import os
import numpy as np

for filename in os.listdir("training"):
    f = open(f"training/{filename}")
    data = f.read()

    dicts = re.findall("\{\"_id.*?file_map.*?}\n", data)

    dicts = dicts[:5]

    arr = np.array(dicts)

    np.save('my_array', arr)




