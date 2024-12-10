struct RNNOutput{T, N, U, V, R} <: AbstractTimeseriesSolution{T, N, U}
    u::U
    t::V
    r::R
end

function RNNOutput(u::Nothing, t::Nothing, r)
    RNNOutput{Nothing, 0, typeof(u), typeof(t), typeof(r)}(u, t, r)
end

function RNNOutput(u::AbstractVector, t, r)
    N = u isa Vector{Vector{T}} where T ? 2 : 1
    T = eltype(eltype(u))
    RNNOutput{T, N, typeof(u), typeof(t), typeof(r)}(u, t, r)
end

struct RNNStates{T, U, V} <: AbstractTimeseriesSolution{T, 2, U}
    u::U
    t::V
end

function RNNStates(u::AbstractVector, t)
    T = eltype(eltype(u))
    RNNStates{T, typeof(u), typeof(t)}(u, t)
end