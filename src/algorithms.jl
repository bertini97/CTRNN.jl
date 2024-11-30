abstract type AbstractAlgorithm end

struct DiscreteDrive <: AbstractAlgorithm end

get_nsteps(alg::DiscreteDrive; u, kvargs...) = size(u, 2)
get_cache(alg::DiscreteDrive, rc; u, kwargs...) = u

struct DiscreteAuto <: AbstractAlgorithm end

get_nsteps(alg::DiscreteAuto; n_steps, kvargs...) = n_steps
function get_cache(alg::DiscreteAuto, rc; kwargs...)
    @assert !isnothing(rc.output.Wo)
    rc.output.Wo * rc.hidden.r
end