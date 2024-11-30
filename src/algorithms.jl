abstract type AbstractAlgorithm end

@inline function standard_step!(int, hidden, input, u)
    r, h, α, Φ = int.r, int.h, hidden.α, hidden.Φ
    put!(h, r, hidden)
    add!(h, u, input)
    @. r = α * Φ.(h) + (1 - α) * r
end

struct DiscreteDrive <: AbstractAlgorithm end

get_nsteps(alg::DiscreteDrive; u, kvargs...) = size(u, 2)
get_cache(alg::DiscreteDrive, rc; u, kwargs...) = u

function perform_step!(int, alg::DiscreteDrive, rc::RC)
    standard_step!(int, rc.hidden, rc.inputs, view(int.cache, :, int.i))
end

struct DiscreteAuto <: AbstractAlgorithm end

get_nsteps(alg::DiscreteAuto; n_steps, kvargs...) = n_steps
function get_cache(alg::DiscreteAuto, rc; kwargs...)
    @assert !isnothing(rc.output.Wo)
    rc.output.Wo * rc.hidden.r
end

function perform_step!(int, alg::DiscreteAuto, rc::RC)
    standard_step!(int, rc.hidden, rc.inputs, int.cache)
    put!(int.cache, rc.output, int.r)
end