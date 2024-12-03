using WeightInitializers
using LinearAlgebra

abstract type AbstractLayer end

struct HiddenLayer{T, V <: AbstractVector{T}, M <: AbstractMatrix{T}} <: AbstractLayer
    α::T
    Φ::Function
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

struct LinearLayer{T, M <: AbstractMatrix{T}} <: AbstractLayer
    W::M
end

@inline function put!(dest, u, layer::LinearLayer)
    mul!(dest, layer.W, u)
end

@inline function add!(dest, u, layer::LinearLayer)
    mul!(dest, layer.W, u, 1.0, 1.0)
end

abstract type AbstractTrainMethod end

struct RidgeRegression{T} <: AbstractTrainMethod
    β::T
end

RidgeRegression() = RidgeRegression(0.0)

ridge_reg(r, y, β) = ((r*r' + β*I)\(r*y'))'

function train(rr::RidgeRegression, r::AbstractMatrix, y::AbstractMatrix)
    Wo = ridge_reg(r, y, rr.β)
    LinearLayer(Wo)
end

function randn_input_layer(N, u::AbstractMatrix{MT}, σi, ::Type{T}=MT) where {T, MT}
    Wi = σi * randn(T, (N, size(u, 1)))
    LinearLayer(Wi)
end

struct RC{I, H, O}
    inputs::I
    hidden::H
    output::O
end

function RC(N::Int, u::AbstractMatrix, y::AbstractMatrix, ::Type{T}=Float64;
            method::AbstractTrainMethod=RidgeRegression(),
            ρ=0.02, Λ=0.8, σi=0.1, σb=1.6, α=0.6, Φ=tanh,
            kwargs...) where T
    hidden = HiddenLayer(N, ρ, Λ, σb, α, Φ, T)
    input = randn_input_layer(N, u, σi, T)
    rc = RC(input, hidden, nothing)
    r = evolve!(rc, DiscreteDrive(), states=true, driver=u)
    output = train(method, r.u, y)

    RC(input, hidden, output)
end