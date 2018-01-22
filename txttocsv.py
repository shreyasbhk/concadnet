import csv
import ast

model_version = 16
run_number = 1

filename = "../Models/"+str(model_version)+"/"+str(run_number)+"/Training_Progress.txt"
ranges_prefix = "../Models/"+str(model_version)+"/"+str(run_number)+"/"


def convert_and_save_ranges(ranges):
    temp_dict = ast.literal_eval(ranges)
    for layer in temp_dict:
        layer_ranges = ast.literal_eval(str(temp_dict[str(layer)]))
        temp_arr = []
        for r in layer_ranges:
            temp_arr.append(layer_ranges[str(r)])
        with open(ranges_prefix+str(layer)+".csv", "a+") as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(temp_arr)


with open(filename) as f:
    lines = f.readlines()
    for line in lines:
        details = line.split(',')
        ranges = ','.join(details[8:58]).replace('\n', '')[15:]
        convert_and_save_ranges(ranges)
