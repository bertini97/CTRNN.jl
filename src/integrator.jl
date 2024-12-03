abstract type AbstractOutputMode end
struct States <: AbstractOutputMode end
struct Output <: AbstractOutputMode end
struct OutputAndStates <: AbstractOutputMode end

mutable struct Integrator{T, V <: Vector{T}, Y, C, O, M}
    n_steps::Int
    i::Int
    t::T
    dt::T
    r::V
    h::V
    y::Y
    cache::C
    out::O
    mode::M
end

function Integrator(alg, rc, output, states; dt, kwargs...)
    r = rc.hidden.r
    h = similar(r)
    n_steps = get_n_steps(alg; kwargs...)
    cache = get_cache(alg, rc; kwargs...)

    if output
        @assert !isnothing(rc.output)
        y_size = size(rc.output.W, 1)
        y = Vector{eltype(r)}(undef, y_size)
        if states
            out = TimeSeriesWithStates(y_size, n_steps, length(r))
            mode = OutputAndStates()
        else
            out = TimeSeries(y_size, n_steps)
            mode = Output()
        end
    else
        y = nothing
        if states
            out = TimeSeries(length(r), n_steps)
            mode = States()
        else
            out = nothing
            mode = nothing
        end
    end

    Integrator(n_steps, 0, 0.0, dt, r, h, y, cache, out, mode)
end

function step!(int, alg, rc, mode::Nothing)
    perform_step!(int, alg, rc)
end

function step!(int, alg, rc, mode::States)
    perform_step!(int, alg, rc)
    put!(int.out, int.i, int.t, int.r)
end

function step!(int, alg, rc, mode::Output)
    y = int.y
    perform_step!(int, alg, rc)
    put!(y, int.r, rc.output)
    put!(int.out, int.i, int.t, y)
end

function step!(int, alg, rc, mode::OutputAndStates)
    @unpack i, r, y = int
    perform_step!(int, alg, rc)
    put!(y, r, rc.output)
    put!(int.out, i, int.t, y)
    put_r!(int.out, i, r)
end

function integration!(int, alg, rc, mode)
    for i in 1:int.n_steps
        int.i = i
        int.t = i * int.dt
        step!(int, alg, rc, mode)
    end
end

function evolve!(rc::RC, alg::AbstractAlgorithm;
                output=false, states=false, dt=0.01, kwargs...)
    int = Integrator(alg, rc, output, states; dt, kwargs...)
    integration!(int, alg, rc, int.mode)
    int.out
end