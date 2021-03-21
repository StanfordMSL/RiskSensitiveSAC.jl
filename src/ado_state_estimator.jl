#///////////////////////////////////////
#// File Name: ado_state_estimator.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/03/20
#// Description: Ado state estimator for RiskSensitiveSAC
#///////////////////////////////////////

using LinearAlgebra

abstract type StateEstimatorParameter end
abstract type StateEstimator end

struct PedestrianKFEstimatorParameter <: StateEstimatorParameter
    Q::Matrix{Float64} # process noise covariance
    R::Matrix{Float64} # position measurement noise covariance
    Σ_init::Matrix{Float64}
    dto::Float64 # discrete time interval [s]
end

mutable struct PedestrianKFEstimator <: StateEstimator
    param::PedestrianKFEstimatorParameter
    μ::Vector{Float64} # mean of [px, py, vx, vy, ax, ay]
    Σ::Matrix{Float64} # cov of [px, py, vx, vy, ax, ay]
end

function PedestrianKFEstimator(param::PedestrianKFEstimatorParameter);
    @assert size(param.Q) == (6, 6);
    @assert size(param.R) == (2, 2);
    @assert size(param.Σ_init) == (6, 6);
    #@assert isposdef(param.Σ_init);
    @assert issymmetric(param.Σ_init);
    μ = zeros(6);
    Σ = copy(param.Σ_init);
    return PedestrianKFEstimator(param, μ, Σ);
end

function initialize!(estimator::PedestrianKFEstimator,
                     pos_measurement::Vector{Float64})
    @assert length(pos_measurement) == 2;
    estimator.μ = [pos_measurement[1], pos_measurement[2], 0.0, 0.0, 0.0, 0.0];
    estimator.Σ = copy(estimator.param.Σ_init);
end

function estimator_predict!(estimator::PedestrianKFEstimator)
    c_1 = estimator.param.dto
    c_2 = 0.5*(estimator.param.dto^2);
    A = [1.0 0.  c_1 0.  c_2 0. ;
         0.  1.0 0.  c_1 0.  c_2;
         0.  0.  1.0 0.  c_1 0. ;
         0.  0.  0.  1.0 0.  c_1;
         0.  0.  0.  0.  1.0 0. ;
         0.  0.  0.  0.  0.  1.0];
    estimator.μ = A*estimator.μ; # constant acceleration model
    estimator.Σ = A*estimator.Σ*(A') + estimator.param.Q;
end

function estimator_update!(estimator::PedestrianKFEstimator,
                 pos_measurement::Vector{Float64})
    @assert length(pos_measurement) == 2;
    C = [1.0 0.  0.  0.  0.  0. ;
         0.  1.0 0.  0.  0.  0. ];
    K = estimator.Σ*(C')/(C*estimator.Σ*(C') + estimator.param.R);
    estimator.μ += K*(pos_measurement - C*estimator.μ);
    estimator.Σ -= K*C*estimator.Σ;
    estimator.Σ = Matrix(Symmetric(estimator.Σ));
end
