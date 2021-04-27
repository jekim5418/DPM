# **D**ynamic **P**ulling **M**ethods(DPM)

This is the official code for papers:
 - [DPM: A Novel Training Method for Physics-Informed Neural Networks in Extrapolation](https://arxiv.org/abs/2012.02681)


We present new algorithm named DPM (Dynamic Pulling Methods). This idea is originated from PINNs(Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.). While observing the capability of PINN as a tool for learning the dynamics of physical processes, we discovered that the accuracy of the approximate solution produced by PINN in an extrapolation setting is significantly reduced compared to that proceed in an extrapolation setting. 
Motivated by this observation, we propose our method to improve the approximation accuracy in extrapolation and to demonstrate the effectiveness of the proposed method with various benchmark problems. 
**This means that our models is well learned governing equation of Parital Differential Equations.**


# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone git@github.com:jekim5418/DPM.git
   ```
 - Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f environment.yml`
   - Manually with pip
     - Set up a Python3 environment (e.g., using virtenv).
     - Install [Tensorflow](tensorflow.org/hub/installation) 1.14
     - Install some other packages:
       ```Shell
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 
       ```
 # Train 
 - If you'd like to train DPM, all of data is in data foler. Just run main model.
 - set hyperparameters that you want(num_layers, num_neurons, learning_rate, limits, epsilon)
   ```Shell
   # Viscous Burgers
   python3 Burgers_parameter_opt_Adam_con_limit_ResNet_DPM.py --num_layers=[num_layers] --num_neurons=[num_neurons] --learning_rate=[learning_rate] --epsilon=[epsilon] --delta=[delta] --w=[w]
   # Inviscid Burgers
   python3 Inviscid_burgers_parameter_opt_Adam_con_limit_ResNet_DPM.py --num_layers=[num_layers] --num_neurons=[num_neurons] --learning_rate=[learning_rate] --epsilon=[epsilon] --delta=[delta] --w=[w]
   # Schrodinger
   python3 Schrodinger_parameter_opt_Adam_con_limit_ResNet_DPM.py --num_layers=[num_layers] --num_neurons=[num_neurons] --learning_rate=[learning_rate]  --epsilon=[epsilon] --delta=[delta] --w=[w]
   # Allen-Cahn
   python3 AC_parameter_opt_Adam_con_limit_ResNet_DPM.py --num_layers=[num_layers] --num_neurons=[num_neurons] --learning_rate=[learning_rate]  --epsilon=[epsilon] --delta=[delta] --w=[w]
   ```
   
  # Evaluation
  - Best hyperparameter is described in below.
  Here are our best hyperparameters of DPM models:

|     Equation     | num_layers | num_neurons | learning rate | epsilon |  delta  |    w    | 
|:----------------:|:----------:|:-----------:|:-------------:|:-------:|:-------:|:-------:|
|  Viscous Burgers |      8     |      20     |     0.005     |  0.001  |   0.08  |  1.001  | 
| Inviscid Burgers |      8     |      20     |     0.01      |  0.0123 |   1.00  |  1.0019 | 
|    Schrodinger   |      3     |      50     |     0.001     |  0.003  |   0.05  |  1.029  | 
|    Allen-Cahn    |      6     |     100     |    0.0005     |  0.001  |   0.01  |  1.022  | 

- For each equation folder, there exists tf_model directory which contains best hyperparameter checkpoints.
  If you want to evaluate our model, then load checkpoints that I saved.
  Also, I saved original-PINN and ResNet-PINN checkpoints in according directory. So it might be possible to compare performance between those two models and ours(DPM).


Comparison of performance in extrapolation between benchmark model and ours(DPM):
- L2-norm error(lower is better):

|     Equation     | Original PINN | ResNet PINN |    **DPM**  |  
|:----------------:|:-------------:|:-----------:|:-----------:|
|  Viscous Burgers |     0.329     |    0.333    |  **0.092**  |
| Inviscid Burgers |     0.131     |    0.095    |  **0.083**  |
|    Schrodinger   |     0.350     |    0.286    |  **0.182**  |
|    Allen-Cahn    |     0.239     |    0.212    |  **0.141**  |

- Explained Variance Score(higher is better):

|     Equation     | Original PINN | ResNet PINN |    **DPM**  |   
|:----------------:|:-------------:|:-----------:|:-----------:|
|  Viscous Burgers |     0.891     |    0.901    |  **0.991**  |
| Inviscid Burgers |     0.214     |    0.468    |  **0.621**  |
|    Schrodinger   |     0.090     |    0.919    |  **0.967**  |
|    Allen-Cahn    |    -4.364     |   -3.902    | **-3.257**  |

- Max Error(lower is better):

|     Equation     | Original PINN | ResNet PINN |    **DPM**  |   
|:----------------:|:-------------:|:-----------:|:-----------:|
|  Viscous Burgers |     0.657     |    1.081    |  **0.333**  |
| Inviscid Burgers |     3.088     |    2.589    |  **1.534**  |
|    Schrodinger   |     1.190     |    1.631    |  **0.836**  |
|    Allen-Cahn    |     4.656     |    4.222    |  **3.829**  |

- Mean Absolute Error(lower is better):

|     Equation     | Original PINN | ResNet PINN |    **DPM**  |   
|:----------------:|:-------------:|:-----------:|:-----------:|
|  Viscous Burgers |     0.085     |    0.108    |  **0.021**  |
| Inviscid Burgers |     0.431     |    0.299    |  **0.277**  |
|    Schrodinger   |     0.212     |    0.142    |  **0.094**  |
|    Allen-Cahn    |     0.954     |    0.894    |  **0.868**  |
  
# Citation
@misc{kim2020dpm,
      title={DPM: A Novel Training Method for Physics-Informed Neural Networks in Extrapolation}, 
      author={Jungeun Kim and Kookjin Lee and Dongeun Lee and Sheo Yon Jin and Noseong Park},
      year={2020},
      eprint={2012.02681},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
  
# Contact
For questions about our paper or code, please contact [Jungeun Kim](jekim5418@yonsei.ac.kr).
