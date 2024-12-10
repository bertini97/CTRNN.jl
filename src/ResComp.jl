#module ResComp

using SciMLBase: AbstractTimeseriesSolution
using RecursiveArrayTools
using LinearAlgebra

include("rnnsolution.jl")
include("rc.jl")
include("algorithms.jl")
include("integrator.jl")

#export RC, RidgeRegression, DiscreteDrive, DiscreteAuto, MatWrap, VecWrap

#export evolve!

#end