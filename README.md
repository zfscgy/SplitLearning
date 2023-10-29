# SplitLearning

This is a simple **split learning** framework, which simulates the split learning procedure in one program while ignoring all the network communication stuff.

Moreover, this code is also the experiment code of the IJCAI'23 paper *[Reducing Communication for Split Learning by Randomized Top-k Sparsification (ijcai.org)](https://www.ijcai.org/proceedings/2023/0519.pdf)* Please cite as

```
@inproceedings{zf2023randtopk,
  title     = {Reducing Communication for Split Learning by Randomized Top-k Sparsification},
  author    = {Zheng, Fei and Chen, Chaochao and Lyu, Lingjuan and Yao, Binhui},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {4665--4673},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/519},
  url       = {https://doi.org/10.24963/ijcai.2023/519},
}
```

## Reproduce the results

The experiment codes are in `examples/task/` directory.

**Notice** that the codes in `examples/sparse/` are legacy scripts and contain errors, just ignore them!
