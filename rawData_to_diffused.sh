#!/bin/bash
python ../Weighted_Graph_Python_GPU/graph_gpu.py -p ./data/position_image.npz -t ./data/texture_image.npz -k 4 -s 1000000000
cp ../Weighted_Graph_Python_GPU/graphs/graph_gpu.npz ./data/
python main.py -i ./data/graph_gpu.npz -t ./data/texture_image.npz -k 4 -it 100
