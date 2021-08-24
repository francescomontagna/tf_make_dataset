import argparse

def parsed_args():
    parser = argparse.ArgumentParser(description="Arguments for dataset processing")
    parser.add_argument('--verbose',  default = False, action="store_true",
        help="set to True when debugging or testing on a new dataset")
    parser.add_argument('--height', type=int, default = 256, 
        help="height for image resizing")
    parser.add_argument('--width', type=int, default = 256, 
        help="width for image resizing")
    parser.add_argument("--path_user", "-u", type=str, 
        help="user to make path where to get and store data")

    return parser.parse_args()