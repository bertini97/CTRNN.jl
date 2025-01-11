using OrdinaryDiffEq, DiffEqCallbacks, RecursiveArrayTools
using LinearAlgebra, Random
using Surrogates
using UnPack
using Lux, LuxCore, WeightInitializers

include("init_weights.jl")

struct RNN{I, H, B, T, F} <: AbstractLuxLayer
    in_dims::Int
    hidden_dims::Int
    init_input::I
    init_hidden::H
    init_bias::B
    τ::T
    Φ::F
end

function LuxCore.initialparameters(rng::AbstractRNG, r::RNN)
    return (input=r.init_input(rng, r.hidden_dims, r.in_dims),
            hidden=r.init_hidden(rng, r.hidden_dims, r.hidden_dims),
            bias=vec(r.init_bias(rng, r.hidden_dims, 1)))
end

function LuxCore.initialstates(rng::AbstractRNG, ::RNN)
    (rng=Lux.Utils.sample_replicate(rng),)
end

#LuxCore.parameterlength(r::RNN) = r.in_dims * r.hidden_dims + r.hidden_dims * r.hidden_dims + r.hidden_dims
LuxCore.statelength(r::RNN) = 1

function (rnn::RNN)(Φₜ::AbstractVector, hₜ::AbstractVector, uₜ, ps)
    copyto!(Φₜ, ps.bias)
    mul!(Φₜ, ps.hidden, hₜ, 1.0, 1.0)
    mul!(Φₜ, ps.input, uₜ, 1.0, 1.0)
    #fast_activation!!(rnn.Φ, Φₜ)
    Φₜ .= rnn.Φ.(Φₜ)
end

struct Output{F} <: AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    Φ::F
end

function LuxCore.initialparameters(rng::AbstractRNG, l::Output)
    return (weight=Nothing,)
end

LuxCore.parameterlength(o::Output) = o.out_dims * o.in_dims
LuxCore.statelength(o::Output) = 0

function (o::Output)(y::AbstractVector, h::AbstractVector, ps)
    mul!(y, ps.weight, h)
    y .= o.Φ(y)
end

function (o::Output)(h::AbstractVector, ps)
    y = ps.weight * h
    o.Φ(y)
end

struct RNNO <: AbstractLuxContainerLayer{(:rnn, :out)}
    rnn::RNN
    out::Output
end

function RNNO(in_dims::Integer, hidden_dims::Integer, out_dims::Integer;
              init_input=truncated_normal(Float64, std=0.1),
              init_hidden=sparse_radius_init(Float64),
              init_bias=truncated_normal(Float64),
              τ=1.0, Φ=tanh)
    rnn = RNN(in_dims, hidden_dims, init_input, init_hidden, init_bias, τ, Φ)
    out = Output(hidden_dims, out_dims, identity)
    RNNO(rnn, out)
end

function dhdt_drive!(dhₜ, hₜ, p, t)
    @unpack rnn, ps_rnn, τ, Φₜ, uₜ, u = p
    u(uₜ, t)
    rnn(Φₜ, hₜ, uₜ, ps_rnn)
    @. dhₜ = (-hₜ + Φₜ)/τ
end

function dhdt_fcast!(dhₜ, hₜ, p, t)
    @unpack rnn, ps_rnn, τ, Φₜ, yₜ, out, ps_out = p
    out(yₜ, hₜ, ps_out)
    rnn(Φₜ, hₜ, yₜ, ps_rnn)
    @. dhₜ = (-hₜ + Φₜ)/τ
end

function save_y(h, t, int)
    p = int.p
    p.out(h, p.ps_out)
end

function drive(rnno::RNNO, h₀::AbstractVector, u,
               tspan, ps, st::NamedTuple, args...;
               save_output=false, save_states=false, kwargs...)
    output = nothing
    callback = nothing
    if save_output
        output = SavedValues(eltype(h₀), typeof(rnno.out(h₀, ps.out)))
        callback = SavingCallback(save_y, output)
    end

    uₜ = u[1] isa Number ? [u[1]] : similar(u[1])
    u(uₜ, tspan[1])

    p = (rnn=rnno.rnn, ps_rnn=ps.rnn, τ=rnno.rnn.τ, Φₜ=similar(h₀), uₜ=uₜ, u=u, out=rnno.out, ps_out=ps.out)
    prob = ODEProblem(dhdt_drive!, h₀, tspan, p)

    h = solve(prob, args...; callback=callback, save_everystep=save_states, kwargs...)
    y = isnothing(output) ? nothing : DiffEqArray(output.saveval, output.t)
    (h, y, st)
end

function forecast(rnno::RNNO, h₀::AbstractVector,
                  tspan, ps, st::NamedTuple, args...;
                  save_output=false, save_states=false, kwargs...)
    output = nothing
    callback = nothing
    if save_output
        output = SavedValues(eltype(h₀), typeof(rnno.out(h₀, ps.out)))
        callback = SavingCallback(save_y, output)
    end
    
    yₜ = rnno.out(h₀, ps.out)
    yₜ = yₜ isa Number ? [yₜ] : yₜ

    p = (rnn=rnno.rnn, out=rnno.out, τ=rnno.rnn.τ, Φₜ=similar(h₀), yₜ=yₜ, ps_rnn=ps.rnn, ps_out=ps.out)
    prob = ODEProblem(dhdt_fcast!, h₀, tspan, p)
    h = solve(prob, args...; callback=callback, save_everystep=save_states, kwargs...)
    y = isnothing(output) ? nothing : DiffEqArray(output.saveval, output.t)
    (h, y, st)
end

function train_ridge(rnno::RNNO, u, y, spinup_tspan, tspan,
                     ps, st, args...; β=1e-7, kwargs...)
    h₀ = zeros(eltype(eltype(u.u)), rnno.rnn.hidden_dims)
    h_spin, _, st = drive(rnno, h₀, u, spinup_tspan, ps, st, args...; kwargs...)
    h, _, st = drive(rnno, h_spin(tspan[1]), u, tspan, ps, st, args...; save_states=true, kwargs...)

    n_samples = min(length(u), length(y))
    t_samples = sample(n_samples, tspan[1], tspan[2], SobolSample())
    #sort!(t_samples)
    h_sampled = VectorOfArray(h.(t_samples))
    y_sampled = VectorOfArray(y.(t_samples))
    h_view = view(h_sampled, :, :)
    adj_y = ndims(y_sampled) == 1 ? y_sampled.u : view(y_sampled, :, :)'

    Wo = ((h_view*h_view' + β*I)\(h_view*adj_y))'
    #y_test = Wo * h_view

    #=fig = Figure()
    ax = Axis(fig[1, 1])
    for i in 1:10
        lines!(ax, t_samples, view(h_sampled, i, :))
    end
    ax2 = Axis(fig[2, 1])
    lines!(ax2, t_samples, adj_y)
    ax3 = Axis(fig[3, 1])
    lines!(ax3, t_samples, y_test')
    display(fig)
    linkxaxes!(ax, ax2)
    linkxaxes!(ax, ax3)=#

    ps = (rnn=ps.rnn, out=(weight=Wo,))
    ps, st
end