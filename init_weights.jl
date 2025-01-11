using Random
using WeightInitializers: PartialFunction

function sparse_radius_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
                              ρ::Number=0.05, Λ::Number=0.8) where {T <: Number}
    weight = sparse_init(rng, T, dims...; sparsity=1.0-ρ)
    λ = maximum(abs.(eigvals(weight)))
    weight .* (Λ / λ)
end

function sparse_radius_init(rng::AbstractRNG, dims::Integer...; kwargs...)
    sparse_radius_init(rng, Float32, dims...; kwargs...)
end

function sparse_radius_init(; kwargs...)
    PartialFunction.Partial{Nothing}(sparse_radius_init, nothing, kwargs)
end

function sparse_radius_init(::Type{T}; kwargs...) where {T <: Number}
    PartialFunction.Partial{T}(sparse_radius_init, nothing, kwargs)
end