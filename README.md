# ResComp.jl

A reservoir computing package for Julia.

## Standard usage

Here is a simple script demonstrating the usage.
```julia
    prob = ODEProblem(lorenz!, [1.0, 0.0, 0.0], (0.0, 1000.0))
    mat = Matrix(solve(prob, dt=dt, saveat=dt, adaptive=false))
    u_train, y_train = @views mat[:, 1:end-1], mat[:, 1+1:end]
    rc = RC(100, u_train, y_train, train_method=RidgeRegression(1e-7))
    evolve!(rc, DiscreteDrive(), driver=u_train)
    y_forecast = evolve!(rc, DiscreteAuto(), outout=true)
```

## Make your own integration algorithm
Declare your custom integration algorithm
```julia
struct CustomAlg <: AbstractAlgorithm end
```
and provide the interface through the following functions.
1. Get the number of steps from the kwargs:
   ```julia
    get_n_steps(alg::CustomAlg; kvargs...)
    ```
2. Get an integration cache (variables to store and retrieve at each step)
   ```julia
   get_cache(alg::CustomAlg, rc::AbstractRC; kwargs...)
   ```
3. Define an integration steps that updates the integrator internal state `int.r`
   ```julia
   perform_step!(int, alg::CustomAlg, rc::AbstractRC)
   ```
## Make your own RC
Declare your custom RC with custom layers
```julia
struct CustomRC{I, C, H, O}
    input::I
    custom::C
    hidden::H
    output::O
end
```
Then create a constructor for it and a custom integrator. If you want to use the standard integrator algorithms make sure it has `input`, `hidden` and `output`.