import tester
import argparse

def main():
    parser = argparse.ArgumentParser(description='EE219 Project 1')
    parser.add_argument('-t','--task', help='define which task to run', required=True)

    args = vars(parser.parse_args())

    tester.startTester(args['task'])

if __name__ == '__main__':
    main()