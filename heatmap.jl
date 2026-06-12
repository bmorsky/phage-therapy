using Roots, LinearAlgebra, DifferentialEquations, DiffEqCallbacks, Plots, Parameters

output_normal = zeros(901,1201)
output_weak = zeros(901,1201)

params = (β = 100, γ = 100.0, δ = 0.002, ϵ = 0.082, ζ = 2.2, η = 0.1, θ = 0.97,
    κ = 1000.0, μ = 0.01, ρ = 1.0, σ = 0.005, ϕ = 0.05, ω = 1.0)

params_weak = (β = 100, γ = 100.0, δ = 0.02, ϵ = 0.082, ζ = 2.2, η = 1.0, θ = 0.01,
    κ = 1000.0, μ = 0.01, ρ = 1.0, σ = 0.005, ϕ = 0.05, ω = 1.0)

# Defining ODE for time series simulation
function system_ode!(du, u, p, t)
    B, I, P = u
    @unpack β, γ, δ, ϵ, ζ, η, θ, κ, μ, ρ, σ, ϕ, ω = p

    du[1] = ρ * B * (1 - B / κ)  - (ϵ*B*I)/(1 + B/ζ) - (ϕ * B * P) / (1 + B / γ)
    du[2] = ((θ * B) / (B + η) - μ * B - δ) * I + σ
    du[3] = β * ϕ * B * P / (1 + B / γ) -ω * P
end

checkdomain(u, p, t) = any(x -> x < 0, u)
affect_P!(integrator) = integrator.u[3] += 1.0
cb_P = PresetTimeCallback(100, affect_P!)
condition_B_eliminated(u, t, integrator) = u[1] - 1e-6
function affect_B_eliminated!(integrator)
    integrator.u[1] = 0.0
    terminate!(integrator)
end
condition_P_eliminated(u, t, integrator) = u[3] - 1e-6
function affect_P_eliminated!(integrator)
    integrator.u[3] = 0.0
end
cb_B_eliminated = ContinuousCallback(condition_B_eliminated,affect_B_eliminated!)
cb_P_eliminated = ContinuousCallback(condition_P_eliminated,affect_P_eliminated!)
cbset = CallbackSet(cb_B_eliminated, cb_P_eliminated)

for i=1:901
    for j=1:1201
        B₀ = 10.0^(0.01*(i-1)-6)
        P₀ = 10.0^(0.01*(j-1)-6)

        # Normal immune system
        prob = ODEProblem(system_ode!, [B₀, params[:σ] / params[:δ], P₀], (0.0, 400.0), params)
        sol = solve(prob, Rodas4P(), isoutofdomain=checkdomain, callback = cbset, saveat=1)

        if any(x -> x < 1e-6, sol[1,:])
            output_normal[i,j] = 1
        end

        # Weakend immune system
        prob = ODEProblem(system_ode!, [B₀, params_weak[:σ] / params_weak[:δ], P₀], (0.0, 400.0), params)
        sol = solve(prob, Rodas4P(), isoutofdomain=checkdomain, callback = cbset, saveat=1)

        if any(x -> x < 1e-6, sol[1,:])
            output_weak[i,j] = 1
        end

    end
end

colors = [RGB(204/255,42/255,54/255), RGB(0,160/255,176/255)]
discrete_colors = cgrad(colors, categorical=true)

heatmap(output, color = discrete_colors, cbar = false,
yticks = ([1, 301, 601, 901],["10⁻⁶","10⁻³","1","10³"]), ylabel="B₀",
xticks = ([1, 301, 601, 901, 1201],["10⁻⁶","10⁻³","1","10³","10⁶"]), xlabel="P₀",
size=(300, 200))
savefig("heatmap_normal.pdf")

heatmap(output_weak, color = discrete_colors, cbar = false,
yticks = ([1, 301, 601, 901],["10⁻⁶","10⁻³","1","10³"]), ylabel="B₀",
xticks = ([1, 301, 601, 901, 1201],["10⁻⁶","10⁻³","1","10³","10⁶"]), xlabel="P₀",
size=(300, 200))
savefig("heatmap_weak.pdf")