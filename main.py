import argparse
from sort import Sorter


def main():
    parser = argparse.ArgumentParser(description='Image sorter')
    parser.add_argument('--input_dir', help='the faces to search for are in this directory\'s images',
                        required=True)
    parser.add_argument('--root_images', help='the images to search in - parent folder of the image folders',
                        required=True)
    args = parser.parse_args()

    if not args.input_dir or not args.root_images:
        raise Exception('Not all parameters given')

    sort = Sorter(args.root_images, args.input_dir)
    sort.sort_wrapper()


if __name__ == '__main__':
    main()
