# Know-Evolve
Implementation code for experiments in ICML '17 paper "[Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs](http://proceedings.mlr.press/v70/trivedi17a/trivedi17a.pdf)"

## Install
To get source code, run:

```
git clone https://github.gatech.edu/rtrivedi6/Know-Evolve.git
```

There are two steps required for complete installation:

1. Install base graphnn library:

```
cd code/graphnn_base
```
Please follow the intallation instructions provided on [Readme](https://github.gatech.edu/rtrivedi6/Know-Evolve/tree/master/code/graphnn_base) page. This is an obsolete and slightly modified version of [graphnn](https://github.com/Hanjun-Dai/graphnn) library.

2. Build Know-Evolve Code:

```
cd code/know_evolve
make
```
This would create a build directory. If you get any error, please check to see that paths in your Makefile are correct.

## Run Experiments

To run experiments on small sample dataset (500 entities):

```
cd code/know_evolve
./run_small.sh
```
To run experiments on full dataset (will require longer testing time):

```
cd code/know_evolve
./run_large.sh
```
You can change various hyper-parameters and try your own dataset using configurations in these files.

#### Contact: rstrivedi AT gatech DOT edu

## Reference
```
@InProceedings{trivedi2017knowevolve,
  title = 	 {Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs},
  author = 	 {Trivedi, Rakshit and Dai, Hanjun and Wang, Yichen and Song, Le},
  booktitle = 	 {Proceedings of the 34th International Conference on Machine Learning},
  year = 	 {2017}
}
```
