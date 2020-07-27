# Code to run the reinforcement learning experiments.


# Dependencies
First make an environment:
```
conda create -n DDQN python=3.7
conda activate DDQN
```
Then install pytorch:
```
pip install torch==1.4.0+cu100 torchvision==0.4.1+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```

Then install the pip packs with:
```
pip install -r requirements.txt
```
To execute DDQN, simply run
```
python DDQN-v5.py
```
The hyper-parameters should be changed in the file itself.
If you want to run SPACE-DDQN or SPACES-DDQN, please also install the dependencies from ../VAE in the same environment as used for RL.
