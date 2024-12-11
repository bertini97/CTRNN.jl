using Distributions
using LinearAlgebra


abstract type AbstractLayer end


struct HiddenLayer{T, F <: Function, V <: AbstractVector{T}, M <: AbstractMatrix{T}} <: AbstractLayer
    α::T
    Φ::F
    r::V
    b::V
    W::M
end

function HiddenLayer(N, ρ, Λ, σb, α, Φ, ::Type{T}=Float64) where T
    r = zeros(T, N)
    b = σb * randn(T, N)
    W = sparse_init(T, N, N; sparsity=1-ρ)
    W = W .* (Λ / maximum(abs.(eigvals(W))))
    HiddenLayer(α, Φ, r, b, W)
end

@inline function put_current!(h, r, hidden::HiddenLayer)
    copyto!(h, hidden.b)
    mul!(h, hidden.W, r, 1.0, 1.0)
end

@inline function put_state!(r, h, hidden)
    α = hidden.α
    @. r = α * hidden.Φ.(h) + (1 - α) * r
end


struct LinearLayer{T, N, A <: AbstractArray{T, N}} <: AbstractLayer
    W::A
end

@inline get_output(u, layer::LinearLayer) = layer.W * u
@inline function put_output!(dest, u, layer::LinearLayer)
    mul!(dest, layer.W, u)
end
@inline function add_current!(dest, u, layer::LinearLayer)
    mul!(dest, layer.W, u, 1.0, 1.0)
end


abstract type AbstractTrainMethod end

struct RidgeRegression{T} <: AbstractTrainMethod
    β::T
end

RidgeRegression() = RidgeRegression(0.0)

function train(rr::RidgeRegression, r::RNNStates, y::AbstractVectorOfArray)
    r = view(r, :, :)
    adj_y = ndims(y) == 1 ? y.u : view(y, :, :)'
    Wo = ((r*r' + rr.β*I)\(r*adj_y))'
    LinearLayer(Wo)
end

function randn_input_layer(N, u, σi, ::Type{T}=Float64) where T
    Wi = σi * randn(T, ndims(u) == 1 ? N : (N, size(u, 1)))
    LinearLayer(Wi)
end


abstract type AbstractRC end


struct RC{I, H, O} <: AbstractRC
    input::I
    hidden::H
    output::O
end

function RC(N::Int, u, y, ::Type{T}=Float64;
            uspin=nothing, method=RidgeRegression(),
            ρ=0.02, Λ=0.8, σi=0.1, σb=1.6, α=0.6, Φ=tanh) where T
    
    hidden = HiddenLayer(N, ρ, Λ, σb, α, Φ, T)
    input = randn_input_layer(N, u, σi, T)
    rc = RC(input, hidden, nothing)

    if !isnothing(uspin)
        evolve!(rc, DiscreteDrive(), driver=uspin)
    end

    sol = evolve!(rc, DiscreteDrive(), driver=u, save_states=true)
    output = train(method, sol.r, y)

    RC(input, hidden, output)
end