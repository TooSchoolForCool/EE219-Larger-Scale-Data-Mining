import sys
import argparse


def read_in(file_path):
    try:
        file = open(file_path, 'r')
    except:
        sys.stderr.write("[ERROR] read_in(): Cannot open file '%s'\n" % file_path)
        exit(1)

    file_content = []

    for line in file:
        file_content.append(line)

    i = 0
    while i < len(file_content):
        line = file_content[i]
        title = line[5:-6]
        print("\t%s" % title)


        line = file_content[i + 7].strip('\n').strip(' ').strip('\t')
        line = [l.strip('\t') for l in line.split(' ') if l]
        for item in line:
            print("\t%s" % item.replace("cluster_", "c")),
        print("")

        for j in range(8, 28):
            line = file_content[i + j].strip('\n').strip(' ').strip('\t')
            line = [l.strip('\t') for l in line.split(' ') if l]
            for item in line:
                print("%s\t" % item),
            print("")


        i += 28
            

    return file_content


def add_parser():
    parser = argparse.ArgumentParser(prog='Compare Evaluation Result')

    parser.add_argument("-s", "--src",
        dest = "src",
        help = "source file path",
        required = True
    )
    parser.add_argument("-d", "--dest",
        dest = "dest",
        help = "destination file path",
    )

    return parser


def main():
    parser = add_parser()
    args = parser.parse_args()

    src_file = read_in(args.src)

if __name__ == '__main__':
    main()