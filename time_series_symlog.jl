using Roots, LinearAlgebra, DifferentialEquations, Plots, Parameters

# y-axis scaling function
function symlog(x)
    if x != NaN
    return log10.(1 .+ 10000*x)
    end
end

T = 400.0
# Parameters
params = (β=100.0, γ=100.0, δ=0.002, ϵ=0.082, ζ=2.2, η=0.1, θ=0.97,
    κ=1000.0, μ=0.01, ρ=1.0, σ=0.005, ϕ=0.05, ω=1.0)

############# Calculate ℰ₁ ############
equilibrium = [0.0 params[:σ] / params[:δ] 0.0]

############# Calculate ℰ₂ ############
# Defining equilibria polynomials
function B_eq!(B, p)

    @unpack β, γ, δ, ϵ, ζ, η, θ, κ, μ, ρ, σ, ϕ, ω = p

    c₄ = μ / (ζ * κ)
    c₃ = μ / κ - μ / ζ + (η * μ + δ - θ) / (ζ * κ)
    c₂ = (δ * η) / (ζ * κ) + (θ - η * μ - δ) * (1 / ζ - 1 / κ) - μ
    c₁ = θ - η * μ - δ - δ * η * (1 / ζ - 1 / κ) + (ϵ * σ) / ρ
    c₀ = (ϵ * (σ / δ) - ρ) * ((δ * η) / ρ)

    return c₄ * B^4 + c₃ * B^3 + c₂ * B^2 + c₁ * B + c₀
end

function I_eq!(B, p)

    @unpack β, γ, δ, ϵ, ζ, η, θ, κ, μ, ρ, σ, ϕ, ω = p

    return (ρ * (1 .- B / κ) .* (1 .+ B/ζ)) / ϵ
end

# Solving for equilibrium numerically
sol_B_eq = find_zeros(x -> B_eq!(x, params), (0, params[:κ]))
sol_I_eq = I_eq!(sol_B_eq, params)
equilibrium = vcat(equilibrium, hcat(sol_B_eq, sol_I_eq, zeros(length(sol_B_eq))))

############# Calculate ℰ₃ ############
B₃ = 1 / ((params[:β] * params[:ϕ]) / params[:ω] - 1 / params[:γ])
I₃ = params[:σ] / (params[:μ] * B₃ + params[:δ] - params[:θ] * B₃ / (B₃ + params[:η]))
P₃ = ((1 + B₃ / params[:γ]) / params[:ϕ]) * (params[:ρ] * (1 - B₃ / params[:κ]) - params[:ϵ] * I₃ / (1 + B₃ / params[:ζ]))

equilibrium = vcat(equilibrium, [B₃ I₃ P₃])

#######################################

# Computing the Jacobian at equilibrium
function jacobian(ℰ, p)
    B, I, P = ℰ
    @unpack β, γ, δ, ϵ, ζ, η, θ, κ, μ, ρ, σ, ϕ, ω = p

    J = zeros(3, 3)
    J[1, 1] = ρ * (1 - 2 * B / κ) - (ϵ * I) / ((1 + B / ζ)^2) - (ϕ * P) / ((1 + B / γ)^2)
    J[1, 2] = -(ϵ * B) / (1 + B / ζ)
    J[1, 3] = -(ϕ * B) / (1 + B / γ)

    J[2, 1] = ((θ * η) / (B + η)^2 - μ) * I
    J[2, 2] = (θ * B) / (B + η) - μ * B - δ
    J[2, 3] = 0.0

    J[3, 1] = (β * ϕ * P) / ((1 + B / γ)^2)
    J[3, 2] = 0.0
    J[3, 3] = (β * ϕ * B) / (1 + B / γ) - ω

    return J
end

for i = 1:size(equilibrium, 1)

    J_eq = jacobian(equilibrium[i, :], params)
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
    @unpack β, γ, δ, ϵ, ζ, η, θ, κ, μ, ρ, σ, ϕ, ω = p

    du[1] = ρ * B * (1 - B / κ)  - (ϵ*B*I)/(1 + B/ζ) - (ϕ * B * P) / (1 + B / γ)
    du[2] = ((θ * B) / (B + η) - μ * B - δ) * I + σ
    du[3] = β * ϕ * B * P / (1 + B / γ) -ω * P
end

# Solving numerically for time series
B₀ = 10.0
I₀ = params[:σ] / params[:δ]
P₀ = 0.0
u₀ = [B₀, I₀, P₀]
tspan = (0.0, T)
prob = ODEProblem(system_ode!, u₀, tspan, params)
affect_P!(integrator) = integrator.u[3] += 1.0
cb_P = PresetTimeCallback(100, affect_P!)
checkdomain(u, p, t) = any(x -> x < 0, u)
sol = solve(prob, Rodas4P(), isoutofdomain=checkdomain, callback = cb_P, abstol=1e-10, reltol=1e-10)

# Generating time series plots
B_vals = sol[1, :]
I_vals = sol[2, :]
P_vals = sol[3, :]
println("Baseline minimum values for t>100")
println("B = ", minimum(B_vals[200:end]))
println("I = ", minimum(I_vals[200:end]))
println("P = ", minimum(P_vals[200:end]))
println("Baseline values for T = $T")
println("B = ", B_vals[end])
println("I = ", I_vals[end])
println("P = ", P_vals[end])

plot_font = "Computer Modern"
plot(sol.t, symlog.(B_vals), legend=false, xlabel="Time (hours)", ylabel="B (10⁶ cells/mL)",
    yticks = (symlog.([0, 0.001, 0.1, 1, 10, 1000]),["0","10⁻³","10⁻¹","1","10","10³"]),
    linewidth=2, linecolor=RGB(79 / 255, 55 / 255, 45 / 255), size=(300, 200))
savefig("B_time_series_baseline.pdf")

plot(sol.t, symlog.(I_vals), legend=false, xlabel="Time (hours)", ylabel="I (10⁶ cells/mL)",
    yticks = (symlog.([0, 0.001, 0.1, 1, 10, 1000]),["0","10⁻³","10⁻¹","1","10","10³"]),
    linewidth=2, linecolor=RGB(79 / 255, 55 / 255, 45 / 255), size=(300, 200))
savefig("I_time_series_baseline.pdf")

plot(sol.t, symlog.(P_vals), legend=false, xlabel="Time (hours/mL)", ylabel="P (10⁶ phages/mL)",
    yticks = (symlog.([0, 0.001, 0.1, 1, 10, 1000,10000]),["0","10⁻³","10⁻¹","1","10","10³","10⁴"]),
    linewidth=2, linecolor=RGB(79 / 255, 55 / 255, 45 / 255), size=(300, 200))
savefig("P_time_series_baseline.pdf")

# Plot phase diagram
Tchaos = 6000.0
B₀ = 1.0
I₀ = params[:σ] / params[:δ]
P₀ = 10.0
u₀ = [B₀, I₀, P₀]
tspanchaos = (0.0, Tchaos)
prob = ODEProblem(system_ode!, u₀, tspanchaos, params)
sol = solve(prob, Rodas4P(), isoutofdomain=checkdomain, abstol=1e-10, reltol=1e-10, saveat=0.001)
transient = 5000000 # number of points to drop
B_vals = sol[1, transient:end]
I_vals = sol[2, transient:end]
P_vals = sol[3, transient:end]

plot3d(
    B_vals, I_vals, P_vals,
    xlabel="B (10⁶ cells/mL)", ylabel="          I (10⁶ cells/mL)", zlabel="P (10⁶ phages/mL)",
    colorbar_title="Time",
    legend=false, linecolor=RGB(79 / 255, 55 / 255, 45 / 255),
    alpha=0.8, size=(800, 800), linewidth=2
)
savefig("baseline_phase_diagram.pdf")

# Immunodeficient host, no antibiotics
params_immunodef = merge(params, (; δ=0.02, η=1.0, θ = 0.01))
B₀ = 10.0
I₀ = params[:σ] / params[:δ]
P₀ = 0.0
u₀ = [B₀, I₀, P₀]
tspan = (0.0, T)
prob = ODEProblem(system_ode!, u₀, tspan, params_immunodef)
sol = solve(prob, Rodas5P(), isoutofdomain=checkdomain, callback=cb_P, abstol=1e-10, reltol=1e-10)

# Generating time series plots
B_vals = sol[1, :]
I_vals = sol[2, :]
P_vals = sol[3, :]
println("Immune deficient minimum values for t>100")
println("B = ", minimum(B_vals))
println("I = ", minimum(I_vals))
println("P = ", minimum(P_vals))

plot_font = "Computer Modern"
plot(sol.t, symlog.(B_vals), legend=false, xlabel="Time (hours)", ylabel="B (10⁶ cells/mL)",
    yticks = (symlog.([0, 0.001, 0.1, 1, 10, 1000]),["0","10⁻³","10⁻¹","1","10","10³"]),
    linewidth=2, linecolor=RGB(79 / 255, 55 / 255, 45 / 255), size=(300, 200))
savefig("B_time_series_immunodef.pdf")

plot(sol.t, symlog.(I_vals), legend=false, xlabel="Time (hours)", ylabel="I (10⁶ cells/mL)",
    yticks = (symlog.([0, 0.001, 0.1, 1, 10, 1000]),["0","10⁻³","10⁻¹","1","10","10³"]),
    linewidth=2, linecolor=RGB(79 / 255, 55 / 255, 45 / 255), size=(300, 200))
savefig("I_time_series_immunodef.pdf")

plot(sol.t, symlog.(P_vals), legend=false, xlabel="Time (hours/mL)", ylabel="P (10⁶ phages/mL)",
    yticks = (symlog.([0, 0.001, 0.1, 1, 10, 1000,10000]),["0","10⁻³","10⁻¹","1","10","10³","10⁴"]),
    linewidth=2, linecolor=RGB(79 / 255, 55 / 255, 45 / 255), size=(300, 200))
savefig("P_time_series_immunodef.pdf")

# Plot phase diagram
params_immunodef = merge(params, (; δ=0.02, η=1.0, θ = 0.01))
Tchaos = 6000.0
B₀ = 10.0
I₀ = params[:σ] / params[:δ]
P₀ = 10.0
u₀ = [B₀, I₀, P₀]
tspanchaos = (0.0, Tchaos)
prob = ODEProblem(system_ode!, u₀, tspanchaos, params_immunodef)
sol = solve(prob, Rodas4P(), isoutofdomain=checkdomain, abstol=1e-10, reltol=1e-10, saveat=0.001)
transient = 5000000 # number of points to drop
B_vals = sol[1, transient:end]
I_vals = sol[2, transient:end]
P_vals = sol[3, transient:end]

plot3d(
    B_vals, I_vals, P_vals,
    xlabel="B (10⁶ cells/mL)", ylabel="          I (10⁶ cells/mL)", zlabel="P (10⁶ phages/mL)",
    colorbar_title="Time",
    legend=false, linecolor=RGB(79 / 255, 55 / 255, 45 / 255),
    alpha=0.8, size=(800, 800), linewidth=2
)
savefig("immunodef_phase_diagram.pdf")

# Antibiotic treatments
dose = 100
dosetimes = collect(100:24:T)
tspan = (0.0, T)
params_antibiotic = (α=0.1247, β=100.0, γ=100.0, δ=0.002, ϵ=0.082, ζ=2.2, η=0.1, θ=0.97,
    κ=1000.0, μ=0.01, ν=0.34657, ξ=18.24, ρ=1.0, σ=0.005, τ=1.416, ϕ=0.05, ω=1.0)

##### Antibiotic-only therapy #####
# Define ODE for time series simulation
function system_ode_antibiotics_only!(du, u, p, t)
    B, I, A = u
    @unpack α, β, γ, δ, ϵ, ζ, η, θ, κ, μ, ν, ξ, ρ, σ, τ, ϕ, ω = p

    du[1] = ρ * B * (1 - B / κ) - (ϵ * B * I) / (1 + B / ζ) - α * B * (abs(A)^τ) / (abs(A)^τ + ξ^τ)
    du[2] = ((θ * B) / (B + η) - μ * B - δ) * I + σ
    du[3] = -ν * A
end
u₀ = [B₀, I₀, 0]
affect!(integrator) = integrator.u[3] += dose
cb = PresetTimeCallback(dosetimes, affect!)
prob = ODEProblem(system_ode_antibiotics_only!, u₀, tspan, params_antibiotic)
sol = solve(prob, Rodas4P(), isoutofdomain=checkdomain, callback=cb, abstol=1e-10, reltol=1e-10)

# Generating time series plots
B_vals = sol[1, :]
I_vals = sol[2, :]
A_vals = sol[3, :]

plot(sol.t, symlog.(B_vals), legend=false, xlabel="Time (hours)", ylabel="B (10⁶ cells/mL)",
    yticks = (symlog.([0, 0.001, 0.1, 1, 10, 1000]),["0","10⁻³","10⁻¹","1","10","10³"]),
    linewidth=2, linecolor=RGB(79 / 255, 55 / 255, 45 / 255), size=(300, 200))
savefig("B_time_series_antibiotic_only.pdf")

plot(sol.t, symlog.(I_vals), legend=false, xlabel="Time (hours)", ylabel="I (10⁶ cells/mL)",
    yticks = (symlog.([0, 0.001, 0.1, 1, 10, 1000]),["0","10⁻³","10⁻¹","1","10","10³"]),
    linewidth=2, linecolor=RGB(79 / 255, 55 / 255, 45 / 255), size=(300, 200))
savefig("I_time_series_antibiotic_only.pdf")

plot(sol.t, symlog.(A_vals), legend=false, xlabel="Time (hours)", ylabel="A",
    yticks = (symlog.([0, 0.01, 0.1, 1, 10,100]),["0","10⁻²","10⁻¹","1","10","10²"]),
    linewidth=2, linecolor=RGB(79 / 255, 55 / 255, 45 / 255), size=(300, 200))
savefig("A_time_series_antibiotic_only.pdf")

##### Phage-antibiotic combo therapy for immunodeficient host #####
params_antibiotic_immunodef = merge(params_antibiotic, (;  δ=0.02, η=1.0, θ = 0.01))
function system_ode_antibiotics!(du, u, p, t)
    B, I, P, A = u
    @unpack α, β, γ, δ, ϵ, ζ, η, θ, κ, μ, ν, ξ, ρ, σ, τ, ϕ, ω = p

    du[1] = ρ * B * (1 - B / κ) - (ϵ * B * I) / (1 + B / ζ) - (ϕ * B * P) / (1 + B / γ) - α * B * (abs(A)^τ) / (abs(A)^τ + ξ^τ)
    du[2] = ((θ * B) / (B + η) - μ * B - δ) * I + σ
    du[3] = β * ϕ * B * P / (1 + B / γ) - ω * P
    du[4] = -ν * A
end
u₀ = [B₀, I₀, 0, 0]
affect_A!(integrator) = integrator.u[4] += dose
cb_A = PresetTimeCallback(dosetimes, affect_A!)
cbset = CallbackSet(cb_A, cb_P)
prob = ODEProblem(system_ode_antibiotics!, u₀, tspan, params_antibiotic_immunodef)
sol = solve(prob, Rodas4P(), isoutofdomain=checkdomain, callback=cbset, abstol=1e-10, reltol=1e-10)

# Generating time series plots
B_vals = sol[1, :]
I_vals = sol[2, :]
P_vals = sol[3, :]
A_vals = sol[4, :]
println("Immune deficient combo therapy minimum values for t>100")
println("B = ", minimum(B_vals))
println("I = ", minimum(I_vals))
println("P = ", minimum(P_vals))

plot(sol.t, symlog.(B_vals), legend=false, xlabel="Time (hours)", ylabel="B (10⁶ cells/mL)",
    yticks = (symlog.([0, 0.001, 0.1, 1, 10, 1000]),["0","10⁻³","10⁻¹","1","10","10³"]),
    linewidth=2, linecolor=RGB(79 / 255, 55 / 255, 45 / 255), size=(300, 200))
savefig("B_time_series_antibiotic_immunodef.pdf")

plot(sol.t, symlog.(I_vals), legend=false, xlabel="Time (hours)", ylabel="I (10⁶ cells/mL)",
    yticks = (symlog.([0, 0.001, 0.1, 1, 10, 1000]),["0","10⁻³","10⁻¹","1","10","10³"]),
    linewidth=2, linecolor=RGB(79 / 255, 55 / 255, 45 / 255), size=(300, 200))
savefig("I_time_series_antibiotic_immunodef.pdf")

plot(sol.t, symlog.(P_vals), legend=false, xlabel="Time (hours)", ylabel="P (10⁶ phages/mL)",
    yticks = (symlog.([0, 0.001, 0.1, 1, 10, 1000,10000]),["0","10⁻³","10⁻¹","1","10","10³","10⁴"]),
    linewidth=2, linecolor=RGB(79 / 255, 55 / 255, 45 / 255), size=(300, 200))
savefig("P_time_series_antibiotic_immunodef.pdf")

plot(sol.t, symlog.(A_vals), legend=false, xlabel="Time (hours)", ylabel="A",
    yticks = (symlog.([0, 0.01, 0.1, 1, 10,100]),["0","10⁻²","10⁻¹","1","10","10²"]),
    linewidth=2, linecolor=RGB(79 / 255, 55 / 255, 45 / 255), size=(300, 200))
savefig("A_time_series_antibiotic_immunodef.pdf")