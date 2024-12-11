abstract type AbstractAlgorithm end
abstract type AbstractDiscreteAlgorithm <: AbstractAlgorithm end
abstract type AbstractAlgorithmCache end


struct DiscreteDrive <: AbstractDiscreteAlgorithm end

struct DiscreteDriveCache{U} <: AbstractAlgorithmCache
    u::U
end

function alg_stuff(::DiscreteDrive, rc; driver::AbstractDiffEqArray)
    tstart = driver.t[1]
    nsteps = length(driver)
    dt = (driver.t[end] - tstart) / nsteps
    tstart, dt, nsteps, DiscreteDriveCache(driver.u)
end

function perform_step!(int, rc, cache::DiscreteDriveCache)
    hidden = rc.hidden
    r = int.r
    h = int.h
    put_current!(h, r, hidden)
    add_current!(h, cache.u[int.iter], rc.input)
    put_state!(r, h, hidden)
    copyto!(int.r, r)
end


struct DiscreteAuto <: AbstractDiscreteAlgorithm end

struct DiscreteAutoCache{Y} <: AbstractAlgorithmCache
    y::Y
end

function alg_stuff(alg::DiscreteAuto, rc; tspan::Tuple{T, T}, dt::T) where T
    tstart, t_finish = tspan
    nsteps = length(tstart:dt:t_finish)
    @assert !isnothing(rc.output.W)
    out = get_output(rc.hidden.r, rc.output)
    y = ndims(out) == 0 ? typeof(out)[out] : out
    tstart, dt, nsteps, DiscreteAutoCache(y)
end

function perform_step!(int, rc, cache::DiscreteAutoCache)
    hidden = rc.hidden
    r = int.r
    h = int.h
    y = cache.y
    put_current!(h, r, hidden)
    add_current!(h, y, rc.input)
    put_state!(r, h, hidden)
    copyto!(int.r, r)
    put_output!(y, int.r, rc.output)
end