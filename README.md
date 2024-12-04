# ResComp.jl

A reservoir computing package for Julia.

## Standard usage

First, get some training data for autonomous mode: the same timeseries shifted by one timestep:
```julia
prob = ODEProblem(lorenz!, [1.0, 0.0, 0.0], (0.0, 1000.0))
mat = Matrix(solve(prob, dt=dt, saveat=dt, adaptive=false))
u_train, y_train = @views mat[:, 1:end-1], mat[:, 1+1:end]
```
Then build the RC (an ESN in this case) from the training data:
```julia
rc = RC(100, u_train, y_train, method=RidgeRegression(1e-7))
```
For now, only `RidgeRegression` is available as training method, but you can implement your own. Then you can evolve it in time using `evolve!` and an appropriate integration algorithm. For now only `DiscreteDrive` and `DiscreteAuto` (to do a discrete map evolution) are implemented, but you can implement your own. Examples:
```julia
y = evolve!(rc, DiscreteDrive(), driver=u, output=true)
y = evolve(rc, DiscreteAuto(), n_steps=100, output=true, states=true)
```
The return type is a `TimeSeries` if `output` or `states` are set to `true`; if both are set, it return a `TimeSeriesWithStates`.

## Make your own training method
Here is the interface:
```julia
struct CustomTrainMethod <: AbstractTrainMethod
train(::CustomTrainMethod, u_train, y_train)
```

## Make your own integration algorithm
The integrator is an object containing all the necessary variables for a time step: the internal state `r`, the integration time `t`, etc. An algorithm usually employs a cache where it stores the additional variables it needs. If it needs more, you need a custom datatype (a struct). The cache is stored in the integrator. Here is the interface
```julia
struct CustomAlg <: AbstractAlgorithm end
get_n_steps(::CustomAlg; kvargs...)
get_cache(::CustomAlg, rc::AbstractRC; kwargs...)
perform_step!(int, ::CustomAlg, rc::AbstractRC)
```

## Make your own RC
Declare your custom RC with custom layers
```julia
struct CustomRC{I, C, H, O} <: AbstractRC
    input::I
    custom::C
    hidden::H
    output::O
end
```
Then create a constructor for it and a custom integrator. If you want to use the standard integrator algorithms make sure it has `input`, `hidden` and `output`.