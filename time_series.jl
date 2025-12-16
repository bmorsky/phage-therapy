using Roots, LinearAlgebra, DifferentialEquations, Plots, Parameters, LaTeXStrings

T = 400.0
# Parameters
params = (β = 100, γ = 10.0, δ = 0.002, ϵ = 0.082, ζ = 2.2, η = 0.1, θ = 0.97,
    κ = 1000.0, μ = 0.01, ρ = 1.0, σ = 0.005, ϕ = 0.05, ω = 1.0)

############# Calculate ℰ₁ ############
equilibrium = [0.0 params[:σ]/params[:δ] 0.0]

############# Calculate ℰ₂ ############
# Defining equilibria polynomials
function B_eq!(B, p)

    @unpack  β, γ, δ, ϵ, ζ, η, θ, κ, μ, ρ, σ, ϕ, ω  = p

    c₄ = μ/(ζ*κ)
    c₃ = μ/κ - μ/ζ + (η*μ+δ-θ)/(ζ*κ)
    c₂ = (δ*η)/(ζ*κ) + (θ - η*μ - δ)*(1/ζ - 1/κ) - μ
    c₁ = θ -η*μ - δ - δ*η*(1/ζ - 1/κ) + (ϵ*σ)/ρ
    c₀ = (ϵ*(σ/δ) - ρ)*((δ*η)/ρ)

    return c₄*B^4 + c₃*B^3 + c₂*B^2 + c₁*B + c₀
end

function I_eq!(B, p)

    @unpack  β, γ, δ, ϵ, ζ, η, θ, κ, μ, ρ, σ, ϕ, ω  = p

    return (ρ*(1 .- B/κ).*(B .+ ζ))/ϵ
end

# Solving for equilibrium numerically
sol_B_eq = find_zeros(x -> B_eq!(x,params), (0,params[:κ]))
sol_I_eq = I_eq!(sol_B_eq, params)
equilibrium = vcat(equilibrium,hcat(sol_B_eq,sol_I_eq,zeros(length(sol_B_eq))))

############# Calculate ℰ₃ ############
B₃ = 1/((p_var[:β]*params[:ϕ])/params[:ω] - 1/params[:γ])
I₃ = p_var[:σ]/(params[:μ]*B₃ + params[:δ] - params[:θ]*B₃/(B₃ + params[:η]))
P₃ = ((1 + B₃/params[:γ])/params[:ϕ])*(params[:ρ]*(1 - B₃/params[:κ]) - params[:ϵ]*I₃/(1 + B₃/params[:ζ]))

equilibrium = vcat(equilibrium,[B₃ I₃ P₃])

#######################################

# Computing the Jacobian at equilibrium
function jacobian(ℰ,p)
    B, I, P = ℰ
    @unpack β, γ, δ, ϵ, ζ, η, θ, κ, μ, ρ, σ, ϕ, ω  = p

    J = zeros(3,3)
    J[1,1] = ρ*(1-2*B/κ) - (ϵ*I)/((1+B/ζ)^2) - (ϕ*P)/((1+B/γ)^2)
    J[1,2] = -(ϵ*B)/(1+B/ζ)
    J[1,3] = -(ϕ*B)/(1+B/γ)
    
    J[2,1] = ((θ*η)/(B+η)^2 - μ)*I
    J[2,2] = (θ*B)/(B+η) - μ*B - δ
    J[2,3] = 0.0
    
    J[3,1] = (β*ϕ*P)/((1+B/γ)^2)
    J[3,2] = 0.0
    J[3,3] = (β*ϕ*B)/(1+B/γ) - ω

    return J
end

for i = 1:size(equilibrium,1)

J_eq = jacobian(equilibrium[i,:],params)
println("\nJacobian at equilibrium:")
println(J_eq)

eigvals_eq = eigvals(J_eq)
println("\nEigenvalues at equilibrium:")
println(eigvals_eq)

if all(real.(eigvals_eq) .< 0)
    println("→ Stable equilibrium (sink)")
elseif all(real.(eigvals_eq) .> 0)
    println("→ Unstable equilibrium (source)")
elseif any(real.(eigvals_eq) .< 0) && any(real.(eigvals_eq) .> 0)
    println("→ Saddle point (unstable)")
else
    println("→ Center or marginally stable")
end
end

# Defining ODE for time series simulation
function system_ode!(du, u, p, t)
    B, I, P = u
    @unpack β, γ, δ, ϵ, ζ, η, θ, κ, μ, ρ, σ, ϕ, ω  = p

    du[1] = ρ*B*(1 - B/κ) - (ϵ*B*I)/(1 + B/ζ) - (ϕ*B*P)/(1 + B/γ)
    du[2] = ((θ*B)/(B + η) - μ*B - δ)*I + σ
    du[3] = β*ϕ*B*P/(1 + B/γ) - ω*P
end

# Solving numerically for time series
B₀ = 100.0
I₀ = params[:σ]/params[:δ]
P₀ = 1.0
u₀ = [B₀, I₀, P₀]
tspan = (0.0, T)
prob = ODEProblem(system_ode!, u₀, tspan, params)
checkdomain(u, p, t) = any(x -> x < 0, u)
sol = solve(prob,Rodas4P(),isoutofdomain=checkdomain,abstol=1e-10,reltol=1e-10)

# Generating time series plots
B_vals = sol[1,:]
I_vals = sol[2,:]
P_vals = sol[3,:]
println(minimum(B_vals))
println(minimum(I_vals))
println(minimum(P_vals))

plot_font = "Computer Modern"
plot(sol.t, B_vals, legend=false, xlabel="Time (hours)", ylabel="B (10⁶ cells/mL)", 
linewidth=2, linecolor=RGB(79/255,55/255,45/255), size=(300,200))
savefig("B_time_series.pdf")

plot(sol.t, I_vals, legend=false, xlabel="Time (hours)", ylabel="I (10⁶ cells/mL)",
linewidth=2, linecolor=RGB(79/255,55/255,45/255), size=(300,200))
savefig("I_time_series.pdf")

plot(sol.t, P_vals, legend=false, xlabel="Time (hours/mL)", ylabel="P (10⁶ phages/mL)",
linewidth=2, linecolor=RGB(79/255,55/255,45/255), size=(300,200))
savefig("P_time_series.pdf")

# Phase Portrait
B_vals = sol[1,:] 
I_vals = sol[2,:]
P_vals = sol[3,:]

# Create 3D plot
plot3d(
    B_vals, I_vals, P_vals,
    line_z = sol.t,
    xlabel="B (10⁶ cells/mL)", ylabel="I (10⁶ cells/mL)", zlabel="P (10⁶ phages/mL)",
    colorbar_title = "Time", linewidth=2,
    legend=false, size=(600,600)
)

# Save figure
savefig("phase_portrait_BIP_baseline.pdf")

# Plot chaotic attractor
Tchaos = 6000.0
B₀ = 100.0
I₀ = params[:σ]/params[:δ]
P₀ = 1.0
u₀ = [B₀, I₀, P₀]
tspanchaos = (0.0, Tchaos)
prob = ODEProblem(system_ode!, u₀, tspanchaos, params)
checkdomain(u, p, t) = any(x -> x < 0, u)
sol = solve(prob,Rodas4P(),isoutofdomain=checkdomain,abstol=1e-10,reltol=1e-10,saveat=0.001)
transient = 5000000 # number of points to drop
B_vals = sol[1, transient:end]
I_vals = sol[2, transient:end]
P_vals = sol[3, transient:end]

plot3d(
    B_vals, I_vals, P_vals,
    xlabel="B (10⁶ cells/mL)", ylabel="I (10⁶ cells/mL)", zlabel="P (10⁶ phages/mL)",
    colorbar_title = "Time",
    legend=false, linecolor=RGB(79/255,55/255,45/255),
    alpha=0.5, size=(600,600), linewidth=2
)
savefig("chaotic_attractor_BIP.pdf")

# Antibiotic treatments
# Define ODE for time series simulation
function system_ode_antibiotics!(du, u, p, t)
    B, I, P, A = u
    @unpack α, β, γ, δ, ϵ, ζ, η, θ, κ, μ, ν, ξ, ρ, σ, τ, ϕ, ω  = p

    du[1] = ρ*B*(1 - B/κ) - (ϵ*B*I)/(1 + B/ζ) - (ϕ*B*P)/(1 + B/γ) - α*B*(A^τ)/(A^τ + ξ^τ)
    du[2] = ((θ*B)/(B + η) - μ*B - δ)*I + σ
    du[3] = β*ϕ*B*P/(1 + B/γ) - ω*P
    du[4] = - ν*A
end
dose = 40
dosetimes = collect(100:24:T)
tspan = (0.0, T)
affect!(integrator) = integrator.u[4] += dose
cb = PresetTimeCallback(dosetimes, affect!)
params_antibiotic = (α = 0.1247, β = 100, γ = 10.0, δ = 0.002, ϵ = 0.082, ζ = 2.2, η = 0.1, θ = 0.97,
    κ = 1000.0, μ = 0.01, ν = 0.34657, ξ = 18.24, ρ = 1.0, σ = 0.005, τ = 1.416, ϕ = 0.05, ω = 1.0)
u₀ = [B₀, I₀, P₀, 0]
prob = ODEProblem(system_ode_antibiotics!, u₀, tspan, params_antibiotic)
checkdomain(u, p, t) = any(x -> x < 0, u)
sol = solve(prob,Rodas4P(),isoutofdomain=checkdomain,callback = cb,abstol=1e-10,reltol=1e-10)

# Generating time series plots
B_vals = sol[1,:]
I_vals = sol[2,:]
P_vals = sol[3,:]
A_vals = sol[4,:]

plot(sol.t, B_vals, legend=false, xlabel="Time (hours)", ylabel="B (10⁶ cells/mL)", 
linewidth=2, linecolor=RGB(79/255,55/255,45/255), size=(300,200))
savefig("B_time_series_antibiotic.pdf")

plot(sol.t, I_vals, legend=false, xlabel="Time (hours)", ylabel="I (10⁶ cells/mL)", 
linewidth=2, linecolor=RGB(79/255,55/255,45/255), size=(300,200))
savefig("I_time_series_antibiotic.pdf")

plot(sol.t, P_vals, legend=false, xlabel="Time (hours)", ylabel="P (10⁶ phages/mL)", 
linewidth=2, linecolor=RGB(79/255,55/255,45/255),size=(300,200))
savefig("P_time_series_antibiotic.pdf")

plot(sol.t, A_vals, legend=false, xlabel="Time (hours)", ylabel="A", 
linewidth=2, linecolor=RGB(79/255,55/255,45/255), size=(300,200))
savefig("A_time_series_antibiotic.pdf")

# Phase Portrait
B_vals = sol[1,:] 
I_vals = sol[2,:]
P_vals = sol[3,:]

# Create 3D plot
plot3d(
    B_vals, I_vals, P_vals,
    line_z = sol.t,
    xlabel="B (10⁶ cells/mL)", ylabel="I (10⁶ cells/mL)", zlabel="P (10⁶ phages/mL)",
    colorbar_title = "Time", size=(600,600),
    linecolor=RGB(79/255,55/255,45/255),
    legend=false, linewidth=2
)

# Save figure
savefig("phase_portrait_BIP_antibiotic.pdf")