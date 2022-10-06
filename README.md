# RumorGraphMAE

Source code for RumorGraphMAE in paper: 

**Graph Representation Learning with Massive Unlabeled Data for Rumor Detection**

## Run

The two training strategies in the paper can be run in the following ways:

```shell script
nohup python main\(pretrain\).py --gpu 0 &
nohup python main\(semisup\).py --gpu 0 &
```

## Dependencies

- [pytorch](https://pytorch.org/) == 1.12.0

- [torch-geometric](https://github.com/pyg-team/pytorch_geometric) == 2.1.0

- [gensim](https://radimrehurek.com/gensim/index.html) == 4.0.1