#!/bin/bash
cd ../Weighted_Graph_Python_GPU/
time python graph_gpu.py -p ../Python_Graph_Heat_Diffusion_Gpu/data/position_image.npz -t ../Python_Graph_Heat_Diffusion_Gpu/data/texture_image.npz -k 4 -s 1000000000
cd -
cp ../Weighted_Graph_Python_GPU/graphs/graph_gpu.npz ./data/
time python main.py -i ./data/graph_gpu.npz -t ./data/texture_image.npz -k 4 -it 100 
