
# sparsenet-pyotrch

A implementation and exploring adversial attacks against Numenta's Sparse CNN paper.

Reference: [How Can We Be So Dense? The Benefits of Using Highly Sparse Representations](https://arxiv.org/abs/1903.11257)

# Train

```
python main.py --save-movel --lr 0.007
```

# Test

IFGSM (Iterative Gradent Sign Method)
```
python ifgsm.py --num-iteration 10 --model-type kwinner
```

Fool Box (Boolbox 3 required)

```
python fb.py --model-type kwinner
```
