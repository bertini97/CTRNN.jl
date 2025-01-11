abstract type AbstractRNNMode end
struct Auto <: AbstractRNNMode end
struct Driven <: AbstractRNNMode end

struct DrivenCache{H, I, O, C, U, D}
    hidden::H
    input::I
    output::O
    h::C
    driver::U
end

get_tspan(::Driven; driver::AbstractDiffEqArray) = (driver.t[1], driver.t[end])

function get_cache(::Driven, rc; driver::AbstractDiffEqArray)
    h = similar(rc.hidden.r)
    DrivenCache(rc.hidden, rc.input, rc.output, h, driver)
end

@inline function update_dr!(dr, r, h, α, Φ)
    @. dr = α * -r + Φ.(h)
end

function rnnscheme!(dr, r, p::DrivenCache, t)
    @unpack hidden, input, h, u = p
    put_current!(h, r, hidden)
    add_current!(h, u(t), input)
    update_dr!(dr, r, h, hidden.α, hidden.Φ)
end

struct AutoCache{H, I, O, C, U}
    hidden::H
    input::I
    output::O
    h::C
    y::U
end

get_tspan(::Auto; tspan) = tspan

function get_cache(::Auto, rc; tspan)
    y = get_output(rc.hidden.r, rc.output)
    h = similar(rc.hidden.r)
    AutoCache(rc.hidden, rc.input, rc.output, h, y)
end

function rnnscheme!(dr, r, p::AutoCache, t)
    @unpack hidden, input, output, h, y = p
    put_output!(y, r, output)
    put_current!(h, r, hidden)
    add_current!(h, y, input)
    update_dr!(dr, r, h, hidden.α, hidden.Φ)
end

struct RNNOutput{O, S}
    output::O
    states::S
end

function evolve!(rc::AbstractRC, mode::AbstractRNNMode, args...;
                 save_output=false, save_states=false, kwargs...)
    if mode isa Auto || save_output
        @assert !isa(rc.output, Nothing)
    end

    cache = get_cache(mode, rc; kwargs...)
    tspan = get_tspan(mode; kwargs...)

    output = nothing
    callback = nothing
    if save_output
        save_y(r, t, int) = get_output(r, int.p.output)
        output = SavedValues(Float64, typeof(get_output(rc.hidden.r, rc.output)))
        SavingCallback(save_y, output)
    end

    prob = ODEProblem(rnn_standard!, rc.hidden.r, tspan, cache)
    states = solve(prob, Euler(), dt=dt, adaptive=false,
                   save_everystep=save_states, callback=callback)
    rc.hidden.r .= states[end]

    RNNOutput(output, save_states ? states : nothing)
end