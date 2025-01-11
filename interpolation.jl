using RecursiveArrayTools
using SciMLBase: AbstractDiffEqInterpolation, LinearInterpolation

function (arr::DiffEqArray)(out, t)
    interpolation!(out, t, arr.u, arr.t)
end

@inline function interpolation!(out, tval::Number, u, t)
    tdir = sign(t[end] - t[1])
    @inbounds i = searchsortedfirst(t, tval, rev = tdir < 0) # It's in the interval t[i-1] to t[i]
    i == 1 && (i += 1)
    dt = t[i] - t[i - 1]
    Θ = (tval - t[i - 1]) / dt
    interpolant!(out, Θ, dt, u[i - 1], u[i])
end

@inline function interpolant!(out, Θ, dt, y₀, y₁)
    Θm1 = (1 - Θ)
    @. out = Θm1 * y₀ + Θ * y₁
end

function (arr::DiffEqArray)(t)
    interpolation(t, arr.u, arr.t)
end

@inline function interpolation(tval::Number, u, t)
    tdir = sign(t[end] - t[1])
    @inbounds i = searchsortedfirst(t, tval, rev = tdir < 0) # It's in the interval t[i-1] to t[i]
    i == 1 && (i += 1)
    dt = t[i] - t[i - 1]
    Θ = (tval - t[i - 1]) / dt
    interpolant(Θ, dt, u[i - 1], u[i])
end

@inline function interpolant(Θ, dt, y₀, y₁)
    Θm1 = (1 - Θ)
    @. Θm1 * y₀ + Θ * y₁
end