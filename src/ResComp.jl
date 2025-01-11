#module ResComp

using LinearAlgebra
using OrdinaryDiffEq
using DiffEqCallbacks
using RecursiveArrayTools
using UnPack

include("rc.jl")
include("evolve.jl")

#export RC, RidgeRegression, DiscreteDrive, DiscreteAuto, MatWrap, VecWrap

#export evolve!

#end