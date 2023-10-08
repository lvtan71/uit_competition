import json
from sklearn.model_selection import train_test_split
import argparse

def split_data(data_path, outfile_train, outfile_test):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = list(data.items())

    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)

    train_data = dict(train_data)
    test_data = dict(test_data)

    with open(outfile_train, "w", encoding="utf-8") as outfile_train:
        json.dump(train_data, outfile_train, ensure_ascii=False, indent=4)

    with open(outfile_test, "w", encoding="utf-8") as outfile_test:
        json.dump(test_data, outfile_test, ensure_ascii=False, indent=4)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--infile', type=str, help="Name of input file")
    parser.add_argument('--outfile_train', type=str, help="Name of output train file")
    parser.add_argument('--outfile_test', type=str, help="Name of output file test")
    args = parser.parse_args()

    split_data(args.infile, args.outfile_train, args.outfile_test)