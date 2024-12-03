module ResComp

using LinearAlgebra
using UnPack

include("rc.jl")
include("timeseries.jl")
include("algorithms.jl")
include("integrator.jl")

export RC, RidgeRegression, DiscreteDrive, DiscreteAuto

export evolve!

end
