using Roots, LinearAlgebra, DifferentialEquations, DiffEqCallbacks, Plots, Parameters


params = (β = 100, γ = 100.0, δ = 0.002, ϵ = 0.082, ζ = 2.2, η = 0.1, θ = 0.97,
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
cbset = CallbackSet(cb_B_eliminated, cb_P_eliminated, cb_P)
cbset2 = CallbackSet(cb_B_eliminated, cb_P_eliminated)

p_var = merge(params,(;δ = params[:δ]*99/10.0))
prob = ODEProblem(system_ode!, [1.0, p_var[:σ] / p_var[:δ], 10], (0.0, 400.0), p_var)
sol = solve(prob, Rodas4P(), isoutofdomain=checkdomain, callback = cbset2, abstol=1e-10, reltol=1e-10, saveat=1)

B_vals = sol[1, :]
I_vals = sol[2, :]
P_vals = sol[3, :]

for i=1:length(B_vals)
    if B_vals[i] < 1e-6
        B_vals[i] = NaN
    end
    if P_vals[i] < 1e-6
        P_vals[i] = NaN
    end
end

plot(log10.(B_vals))