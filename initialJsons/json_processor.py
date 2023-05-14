import json
import os
import numpy as np

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

json_name = 1
for ix in range(1):
    for f in np.load('my_array2.npy'):
        invoice = json.loads(f)
        file_map = json.loads(invoice["file_map"])
        try:
            file_learnedByDD = invoice["learnedByDD"]
        except Exception as ex:
            file_learnedByDD = {}
        try:
            learnedByVal = invoice["learnedByVal"]
        except Exception as ex:
            learnedByVal = {}
        keys = file_map.keys()
        for key in keys:
            data = []
            file_map_data = file_map[key].copy()
            for i in range(len(file_map[key]['data'])):
                if len(data) == 125:
                    file_map_data['data'] = data
                    json_object = json.dumps({**file_map_data, **{"learnedByDD": file_learnedByDD}, **{"learnedByVal": learnedByVal}},indent=4)
                    with open("created/" + str(json_name) + ".json", "w") as outfile:
                        outfile.write(json_object)
                    json_name = json_name + 1
                    data = data[50:]
                data.append(file_map[key]['data'][i])
            file_map_data['data'] = data
            json_object = json.dumps({**file_map_data, **{"learnedByDD": file_learnedByDD}, **{"learnedByVal": learnedByVal}}, indent=4)
            with open("created/"+str(json_name)+".json", "w") as outfile:
                outfile.write(json_object)
            json_name = json_name + 1

    for file in os.listdir("created"):
        try:
            with open(f'created/{file}', 'r+') as f:
                data = json.load(f)
                dataNew = []
                learnedByDD_words = data.get('learnedByDD')
                learnedByVal_words = data.get('learnedByVal')
                repeted_inf = []
                for key_word in learnedByDD_words:
                    repeted_inf.append((learnedByDD_words[key_word]["value"], key_word))
                for key_word in learnedByVal_words:
                    repeted_inf.append((learnedByVal_words[key_word]["value"], key_word))
                width = data.get('width')
                height = data.get('height')
                for i, word in enumerate(data["data"]):
                    if i==511:
                        json_name = json_name + 1
                        with open("created/" + str(json_name) + ".json", "w") as outfile:
                            outfile.write(json_object)
                    replace_form = []
                    boundingPoly = word.get('boundingPoly')
                    boundingPoly = [boundingPoly["x"],
                                    boundingPoly["y"]-boundingPoly["h"],
                                    boundingPoly["x"]+boundingPoly["w"],
                                    boundingPoly["y"]]
                    checker = normalize_box(boundingPoly, width, height)
                    if any(i > 1000 or i < 0 for i in checker):
                        continue
                    text = word.get('text')
                    if text in [i[0] for i in repeted_inf]:
                        label = [item for item in repeted_inf if text in item][0][1]
                    else:
                        label = "OTHER"
                    replace_form.append(
                        {"text": text, "box": boundingPoly, "label": label, "words": [{"box": boundingPoly, "text": text}]})
                    for sub_form in replace_form:
                        dataNew.insert(0, sub_form)
                data['data'] = dataNew
                data['form'] = data.pop('data')
                f.seek(0)
                f.write(json.dumps(data))
                f.truncate()
        except Exception as ex:
            continue