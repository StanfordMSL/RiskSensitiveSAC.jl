using Distributions
using LinearAlgebra

# Scene loader & Predictor parameters
scene_mode = "synthetic";                                                           # "synthetic" or "data"
prediction_mode = "gaussian";                                                       # "gaussian" or "trajectron" or "oracle"
prediction_device = "cpu";                                                          # "cpu" or "cuda"
prediction_steps = 12;                                                              # number of steps to look ahead in the future
ado_pos_init_dict = Dict("PEDESTRIAN/1" => [0.0, -5.0]);                            # initial ado positions [x, y] [m]
ado_vel_dict = Dict("PEDESTRIAN/1" => MvNormal([0.0, 1.0], Diagonal([0.01, 0.15])));# ado velocity distributions
dto = 0.4;                                                                          # observation update time interval [s]
prediction_rng_seed = 1;                                                            # random seed for prediction (and stochastic transition for "synthetic" scenes)
deterministic = false;                                                              # if true, a single, deterministic sample is drawn regardless of random seed. (num_samples = 1 is needed)
num_samples = 30;                                                                   # number of trajectory samples (per ado agent)
# Cost Parameters
include("params_cost.jl")
# Control Parameters
include("params_control.jl")
# Ego initial state
ego_pos_init_vec = [-5., 0.];                                                       # initial ego position [x, y] [m]
ego_pos_goal_vec = [5., 0.];                                                        # goal ego position [x, y] [m]
# Other parameters
pos_error_replan = 5.0;                                                             # position error for replanning target trajectory [m]
target_speed = 1.0;                                                                 # target speed [m/s]
sim_horizon = 10.0;                                                                 # simulation time horizon [s]
