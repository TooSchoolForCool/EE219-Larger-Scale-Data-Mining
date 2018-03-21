# Project 5: Popularity Prediction on Twitter

This repo is for "EE219 Larger Scale Data Mining" course projects at [UCLA](http://www.ucla.edu).



**Author Info:**

|     Name     |      Contact      |
| :----------: | :---------------: |
|  Zeyu Zhang  | zeyuz@outlook.com |
| Yunchu Zhang |      private      |



## Install Dependencies

This project is develop under `Ubuntu 16.04`  (`MAC OS` is fine) with `Python3.5`. If you do NOT have a Python 3, please install Python 3 first. How to install Python 3 ? Please google it! 



Once you have the Python 3, several things need to be done before executing our code. However, we wrote a simple script to ease your work, it will install the dependencies automatically. Open your terminal, get into the Project-1 directory, and enter following command.

```bash
sh ./install.sh
```



## How to Run?

Before you run our program, we need to have dataset at the **correct** place.

The folder structure is like following

EE219-UID..../

---src/

---tweet_data/

---test_data/

`src`, `test_data`, and `tweet_data` are folders in directory EE219-UID.... The `src` folder contains our source code, the `test_data` folder contains our test data, and the `tweet_data` contains the hashtag tweets data.



**Note:** Before you run our program, you need to run the ***preprocess*** program first. Get into the `src` directory and try following command,

```
python3 preprocess.py
```

After preprocessing is done, you can run our program now.



Open your terminal, get into the `src ` directory where we placed our source code. Try following command to run `--task 1.1`, if you want to execute other task, just change `--task 1.1` to the corresponding task. For example, if you want to try `--task 1.2` just change it to `--task 1.2`

```bash
python3 main.py --task 1.1
```

Note: Available options contain

- --task 1.1
- --task 1.2
- --task 1.3
- --task 1.4
- --task 1.5
- --task 2
- --task 3.1
- --task 3.2