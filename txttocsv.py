import csv
import ast
import os

model_version = 15
run_number = 1

filename = "../Models/"+str(model_version)+"/"+str(run_number)+"/Training_Progress.txt"
csv_filename = "../Models/"+str(model_version)+"/"+str(run_number)+"/Training_Progress.csv"
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


def convert_and_save_details(details):
    temp_arr = [details[0][7:], details[1][7:], details[2][16:], details[3][15:], details[4][18:],
                details[5][17:], details[6][15:], details[7][14:]]
    print(temp_arr)
    with open(csv_filename, "a+") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(temp_arr)


for i in range(5):
    n = ranges_prefix+"Layer "+str(i)+".csv"
    if os.path.isfile(n):
        os.remove(n)

if os.path.isfile(csv_filename):
    os.remove(csv_filename)

with open(filename) as f:
    lines = f.readlines()
    for line in lines:
        details = line.split(',')
        ranges = ','.join(details[8:58]).replace('\n', '')[15:]
        convert_and_save_ranges(ranges)
        print(details[:8])
        convert_and_save_details(details[:8])
