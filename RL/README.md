# Code to run the reinforcement learning experiments.


# Dependencies
First make an environment:
```
conda create -n DDQN python=3.7
conda activate DDQN
```
Then install pytorch:
```
pip install torch==1.4.0+cu100 torchvision==0.4.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```

Then install the pip packs with:
```
pip install -r requirements.txt
```
execute DDQN.py . The hyper-parameters should be changed in the file itself.
