import tasks
import argparse


def main():
    parser = argparse.ArgumentParser(description='EE219 Project 2')

    parser.add_argument('-t','--task',
        dest = "task",
        help = 'define which task to be executed',
        required = True
    )

    args = parser.parse_args()

    tasks.run_task(args.task)


if __name__ == '__main__':
    main()