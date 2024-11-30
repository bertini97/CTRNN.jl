module ResComp

using LinearAlgebra
using UnPack

include("rc.jl")
include("timeseries.jl")
include("algorithms.jl")
include("integrator.jl")

export evolve!

export DiscreteDrive, DiscreteAuto

end
