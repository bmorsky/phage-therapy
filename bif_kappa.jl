using Roots, LinearAlgebra, DifferentialEquations, Plots, Parameters, LaTeXStrings

vars = zeros(100)
eq_B = fill(NaN, 100,5)
eq_I = fill(NaN, 100,5)
eq_P = fill(NaN, 100,5)
eq_stab = fill("unknown", 100, 5)

# Parameters
params = (β = 100, γ = 10.0, δ = 0.002, ϵ = 0.082, ζ = 2.2, η = 0.1, θ = 0.97,
    κ = 1000.0, μ = 0.01, ρ = 1.0, σ = 0.005, ϕ = 0.05, ω = 1.0)

function B_eq!(B, p)
    @unpack β, γ, δ, ϵ, ζ, η, θ, κ, μ, ρ, σ, ϕ, ω  = p
    
    c₄ = μ/(ζ*κ)
    c₃ = μ/κ - μ/ζ + (η*μ+δ-θ)/(ζ*κ)
    c₂ = (δ*η)/(ζ*κ) + (θ - η*μ - δ)*(1/ζ - 1/κ) - μ
    c₁ = θ -η*μ - δ - δ*η*(1/ζ - 1/κ) + (ϵ*σ)/ρ
    c₀ = (ϵ*(σ/δ) - ρ)*((δ*η)/ρ)

    return c₄*B^4 + c₃*B^3 + c₂*B^2 + c₁*B + c₀
end

function I_eq!(B, p)
    @unpack β, γ, δ, ϵ, ζ, η, θ, κ, μ, ρ, σ, ϕ, ω  = p

    return (ρ*(1 .- B/κ).*(1 .+ B/ζ))/ϵ
end

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

function stability(eigvals_eq)
    if all(real.(eigvals_eq) .< 0)
        return "stable"
    elseif any(real.(eigvals_eq) .< 0) && any(real.(eigvals_eq) .> 0)
        return "saddle"
    elseif any(real.(eigvals_eq) .> 0)
        return "unstable"
    else
        return "unknown"
    end
end

for i = 1:100
    p_var = merge(params,(;κ = params[:κ]*i/10.0))
    vars[i] = p_var[:κ]

    ############# Calculate ℰ₁ ############
    eq1 = [0.0 p_var[:σ]/p_var[:δ] 0.0]
    J_eq = jacobian(eq1, p_var)
    eq_B[i,1] = 0.0
    eq_I[i,1] = p_var[:σ]/p_var[:δ]
    eq_P[i,1] = 0.0
    eq_stab[i,1] = stability(eigvals(J_eq))

    ############# Calculate ℰ₂ ############
    local sol_B_eq = find_zeros(x -> B_eq!(x,p_var), (0,params[:κ]+1), atol=1e-10 ,rtol=1e-10 ,xatol=1e-10, xrtol=1e-10)
    local sol_I_eq = I_eq!(sol_B_eq, p_var)
    local equilibrium = hcat(sol_B_eq,sol_I_eq,zeros(length(sol_B_eq)))
    for j = 1:size(equilibrium,1)
        eq_B[i,j+1] = equilibrium[j,1]
        eq_I[i,j+1] = equilibrium[j,2]
        eq_P[i,j+1] = equilibrium[j,3]
        J_eq = jacobian(equilibrium[j,:], p_var)
        eq_stab[i,j+1] = stability(eigvals(J_eq))
    end

    ############# Calculate ℰ₃ ############
    local B₃ = 1/((p_var[:β]*p_var[:ϕ])/p_var[:ω] - 1/p_var[:γ])
    local I₃ = p_var[:σ]/(p_var[:μ]*B₃ + p_var[:δ] - p_var[:θ]*B₃/(B₃ + p_var[:η]))
    local P₃ = ((1 + B₃/p_var[:γ])/p_var[:ϕ])*(p_var[:ρ]*(1 - B₃/p_var[:κ]) - p_var[:ϵ]*I₃/(1 + B₃/p_var[:ζ]))
    if B₃ > 0 &&  I₃ > 0 && P₃ > 0
        eq_B[i,5] = B₃
        eq_I[i,5] = I₃
        eq_P[i,5] = P₃
        J_eq = jacobian([B₃ I₃ P₃], p_var)
        eq_stab[i,5] = stability(eigvals(J_eq))
    end
end

function symlog(x)
    if x != NaN
    return log10.(1 .+ 10000*x)
    end
end
plot_font = "Computer Modern"
color_map = Dict("stable" => RGB(0,160/255,176/255), "unstable" => RGB(235/255,104/255,65/255), "saddle" => RGB(204/255,42/255,54/255), "unknown" => RGB(79/255,55/255,45/255))
marker_colors = map(c -> color_map[c], eq_stab)
scatter(vars,symlog.(eq_B), markercolor=marker_colors,
yticks = (symlog.([0, 0.001, 0.1, 1, 10, 1000,10000]),["0","10⁻³","10⁻¹","1","10","10³","10⁴"]),
markerstrokewidth=0, linewidth=2, colorbar=false, legend=false,
xlabel="carrying capacity, κ",ylabel="B (10⁶ cells/mL)",size=(300,200))

savefig("bif_kappa.pdf")