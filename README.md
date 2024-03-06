## Python Run with GPUs

1 GPU:

```python
python ./main.py --device cuda --batch_size 256 --num_devices 1 --data_dir ./data/256/images/25/256_1/train --test_dir ./data/256/images/25/256_1/test
```

2 GPUs:

```python
python ./main.py --device cuda --batch_size 512 --num_devices 2 --data_dir ./data/256/images/25/256_1/train --test_dir ./data/256/images/25/256_1/test
```

4 GPUs:

```python
python ./main.py --device cuda --batch_size 1024 --num_devices 4 --data_dir ./data/256/images/25/256_1/train --test_dir ./data/256/images/25/256_1/test
```
