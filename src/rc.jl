using WeightInitializers
using LinearAlgebra

abstract type AbstractLayer end
abstract type AbstractInputLayer <: AbstractLayer end
abstract type AbstractOutputLayer <: AbstractLayer end

struct HiddenLayer{T, F, V, M} <: AbstractLayer
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

function put!(dest, r, hidden::HiddenLayer)
    dest .= hidden.b
    mul!(dest, hidden.W, r, 1.0, 1.0)
end

struct LinearOutputLayer{M} <: AbstractOutputLayer
    Wo::M
end

@inline function put!(dest, output::LinearOutputLayer, r)
    mul!(dest, output.Wo, r)
end

abstract type AbstractTrainingMethod end

struct RidgeRegression{T}
    β::T
end

RidgeRegression() = RidgeRegression(0.0)

ridge_regression(r, y, β) = ((r*r' + β*I)\(r*y'))'

function LinearOutputLayer(rr::RidgeRegression, r::AbstractMatrix, y::AbstractMatrix)
    Wo = ridge_regression(r, y, rr.β)
    LinearOutputLayer(Wo)
end

struct LinearInputLayer{T, M <: Matrix{T}} <: AbstractInputLayer
    Wi::M
end

function LinearInputLayer(N, u::AbstractMatrix{MT}, σi, ::Type{T}=MT) where {T, MT}
    Wi = σi * randn(T, (N, size(u, 1)))
    LinearInputLayer(Wi)
end

@inline function add!(dest, u, input::LinearInputLayer)
    mul!(dest, input.Wi, u, 1.0, 1.0)
end

struct RC{I, H, O}
    inputs::I
    hidden::H
    output::O
end

function RC(N::Int, u::AbstractMatrix, y::AbstractMatrix, ::Type{T}=Float64;
            train_method=RidgeRegression(),
            ρ=0.02, Λ=0.8, σi=0.1, σb=1.6, α=0.6, Φ=tanh,
            kwargs...) where T
    hidden = HiddenLayer(N, ρ, Λ, σb, α, Φ, T)
    input = LinearInputLayer(N, u, σi, T)
    rc = RC(input, hidden, nothing)
    r = evolve!(rc, DiscreteDrive(), states=true, u=u)
    output = LinearOutputLayer(train_method, r.u, y)

    RC(input, hidden, output)
end