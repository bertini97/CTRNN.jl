abstract type AbstractTimeSeries{T} <: AbstractMatrix{T} end

struct TimeSeries{T, V <: Vector{T}, M <: Matrix{T}}
       <: AbstractTimeSeries{T}
    t::V
    u::M
end

struct TimeSeriesWithStates{T, V <: Vector{T}, M <: Matrix{T}}
       <: AbstractTimeSeries{T}
    t::V
    u::M
    r::M
end

@inline Base.size(ts::AbstractTimeSeries) = size(ts.u)
@inline Base.getindex(ts::AbstractTimeSeries, i::Int, j::Int) = ts.u[i, j]

function default_members(n_dims::Int, n_steps::Int, ::Type{T}) where T
    Vector{T}(undef, n_steps), Matrix{T}(undef, (n_dims, n_steps))
end

function TimeSeries(n_dims::Int, n_steps::Int, ::Type{T}=Float64) where T
    t, u = default_members(n_dims, n_steps, T)
    TimeSeries(t, u)
end

function TimeSeriesWithStates(n_dims::Int, n_steps::Int, ::Type{T}=Float64) where T
    t, u = default_members(n_dims, n_steps, T)
    TimeSeriesWithStates(t, u, nothing)
end

function TimeSeriesWithStates(n_dims::Int, n_steps::Int, n_nodes::Int, ::Type{T}=Float64) where T
    t, u = default_members(n_dims, n_steps, T)
    r = Matrix{T}(undef, (n_nodes, n_steps))
    TimeSeriesWithStates(t, u, r)
end

@inline function put!(ts::AbstractTimeSeries, i::Int, t, val)
    ts.t[i] = t
    view(ts.u, :, i) .= val
end

@inline function put_r!(tsws::TimeSeriesWithStates, i::Int, val)
    view(tsws.r, :, i) .= val
end