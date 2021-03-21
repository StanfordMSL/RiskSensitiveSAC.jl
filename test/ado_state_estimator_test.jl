#///////////////////////////////////////
#// File Name: ado_state_estimator_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/03/20
#// Description: Test code for src/ado_state_estimator.jl
#///////////////////////////////////////

using LinearAlgebra;

@testset "Pedestrian Kalman Filter Test" begin
    Q = diagm([0.0, 0.0, 0.5, 0.5, 1.0, 1.0]);
    R = diagm([0.0, 0.0]);
    Σ_init = diagm([0.0, 0.0, 3.0, 3.0, 3.0, 3.0]);
    dto = 1.0;

    estimator_param = PedestrianKFEstimatorParameter(Q, R, Σ_init, dto)
    estimator = PedestrianKFEstimator(estimator_param);
    @test estimator.μ == zeros(6);
    @test estimator.Σ == Σ_init;

    pos_array = [ii*ones(2) for ii = 1 : 10];
    initialize!(estimator, pos_array[1]);
    @test estimator.μ == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    @test estimator.Σ == Σ_init;
    for ii = 2 : 10
        estimator_predict!(estimator);
        estimator_update!(estimator, pos_array[ii])
        @test all(estimator.μ[1:2] .== pos_array[ii])
    end
    @test all(isapprox.(estimator.μ[3:4], [1.0 ,1.0], atol=1e-5));
    @test all(isapprox.(estimator.μ[5:6], [0.0, 0.0], atol=1e-5));
end
