conda create -n tf215_env python=3.9 -y
conda activate tf215
+ pip install tensorflow[and-cuda]==2.15.1
+ pip install pandas
+ pip install scikit-learn
+ pip install matplotlib

+ pip install ipykernel
+ python -m ipykernel install --user --name tf215_env --display-name "Python 3 (tf215_env)"
