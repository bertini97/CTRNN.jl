mutable struct Integrator{T, C, S}
    nsteps::Int64
    iter::Int64
    t::T
    dt::T
    tstart::T
    r::Vector{T}
    h::Vector{T}
    cache::C
    sol::S
end

function Integrator(rc, alg, save_output, save_states; kwargs...)
    r = rc.hidden.r
    h = similar(r)
    tstart, dt, nsteps, cache = alg_stuff(alg, rc; kwargs...)
    
    if save_output
        @assert !(rc.output.W isa Nothing)
        y = get_output(r, rc.output)
        times = eltype(r)[]
        output = typeof(y)[]
    else
        times = nothing
        output = nothing
    end
    states = save_states ? RNNStates(typeof(r)[], eltype(r)[]) : nothing
    sol = RNNOutput(output, times, states)

    Integrator(nsteps, 0, tstart, dt, tstart, r, h, cache, sol)
end

function save_step!(int, rc)
    sol = int.sol
    if !(sol.u isa Nothing)
        push!(sol.t, int.t)
        push!(sol.u, get_output(int.r, rc.output))
    end
    if !(sol.r isa Nothing)
        r = sol.r
        push!(r.t, int.t)
        push!(r.u, copy(int.r))
    end
end

function integration!(int, rc, ::AbstractDiscreteAlgorithm)
    for i in 1:int.nsteps
        int.iter = i
        int.t = int.tstart + (i-1) * int.dt
        perform_step!(int, rc, int.cache)
        save_step!(int, rc)
    end
    rc.hidden.r .= int.r
    int.sol
end


function evolve!(rc::AbstractRC, alg::AbstractDiscreteAlgorithm;
                save_output=false, save_states=false, kwargs...)
    int = Integrator(rc, alg, save_output, save_states; kwargs...)
    integration!(int, rc, alg)
end