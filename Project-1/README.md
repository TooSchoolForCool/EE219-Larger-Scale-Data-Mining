# Project 1 Classification Analysis on Textual Data

This repo is for "EE219 Larger Scale Data Mining" course projects at [UCLA](http://www.ucla.edu).



**Author Info:**

|     Name     |      Contact      |
| :----------: | :---------------: |
|  Zeyu Zhang  | zeyuz@outlook.com |
| Yunchu Zhang |      private      |



## Project Description

1. Data Balance

   - Down Sampling
   - Add weight

2. TFxIDF
3. TFxICF
4. Feature Selection (Dimension Reduction)

   - LSI
   - NMF
5. SVM (soft & hard margin) (2-class)
6. Find best param $\gamma$ (by using 5-fold cross-validation)
7. NaÏve Bayes (2-class)
8. Logistic Regression (2-class)
9. Norm Regularization
10. Multiclass Classification (SVM & Naïve Bayes)




## Install Dependencies

This project is develop under `Ubuntu 16.04`  (`MAC OS` is fine) with `Python2`. If you do NOT have a Python2, please install Python 2 first. How to install Python 2 ? Please google it! 



Once you have the Python 2, several things need to be done before executing our code. However, we wrote a simple script to ease your work, it will install the dependencies automatically. Open your terminal, get into the Project-1 directory, and enter following command.

```bash
sh ./install.sh
```



## How to Run?

Open your terminal, get into the`src `directory where we placed our source code. Try following command to run `task a`, if you want to execute other task, just change `a` to the corresponding task. For example, if you want to try `task j` just change it to `--task j`. (Note: only `a to j`  are supported)

```bash
python main.py --task a
```

