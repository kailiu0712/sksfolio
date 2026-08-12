module MarkowitzCommercial

# Shared Julia/JuMP implementation used by all solver adapters.

using JSON3
using JuMP
using LinearAlgebra
using Mmap
using SparseArrays

export MarkowitzInstance
export SolverOptions
export BinarySolverOptions
export load_bundle
export solve_bundle
export solve_instance
export solve_binary_bundle
export solve_binary_instance
export write_result
export main

const MOI = JuMP.MOI
const BUILTIN_SOLVERS = Dict(
    "gurobi" => (
        package = :Gurobi,
        constructor = :Optimizer,
        profile = :gurobi,
        conic = true,
    ),
    "mosek" => (
        package = :Mosek,
        constructor = :Optimizer,
        profile = :mosek,
        conic = true,
    ),
    "mosektools" => (
        package = :Mosek,
        constructor = :Optimizer,
        profile = :mosek,
        conic = true,
    ),
    "clarabel" => (
        package = :Clarabel,
        constructor = :Optimizer,
        profile = :clarabel,
        conic = true,
    ),
    "cosmo" => (
        package = :COSMO,
        constructor = :Optimizer,
        profile = :cosmo,
        conic = true,
    ),
    "highs" => (
        package = :HiGHS,
        constructor = :Optimizer,
        profile = :highs,
        conic = false,
    ),
)

struct OptimizerSpec
    label::String
    package::Symbol
    constructor::Symbol
    profile::Symbol
    conic::Bool
end

"""
Data for the continuous perspective relaxation

    min 0.5 * ||B' * x||^2
        + 0.5 * perspective_weight * sum(t)
        - return_reward * mu' * x

subject to a full-investment constraint, the generic rows of `C`, and
the perspective constraints. `B` is always stored with shape `(d, r)`;
the dense covariance matrix `B * B'` is never formed.
"""
struct MarkowitzInstance{
    TB <: AbstractMatrix{Float64},
    TC <: AbstractMatrix{Float64},
}
    B::TB
    mu::Vector{Float64}
    C::TC
    lower::Vector{Float64}
    upper::Vector{Float64}
    anchor::Union{Nothing, Vector{Float64}}
    k::Int
    perspective_weight::Float64
    return_reward::Float64
    metadata::Dict{String, Any}
end

Base.@kwdef struct SolverOptions
    threads::Int = 0
    tolerance::Float64 = 1.0e-6
    time_limit::Union{Nothing, Float64} = nothing
    log::Bool = false
    warm_start::Bool = true
    initial_x::Union{Nothing, Vector{Float64}} = nothing
    optimizer_attributes::Dict{String, Any} = Dict{String, Any}()
end

Base.@kwdef struct BinarySolverOptions
    threads::Int = 0
    tolerance::Float64 = 1.0e-8
    relative_gap::Float64 = 1.0e-4
    absolute_gap::Float64 = 0.0
    time_limit::Union{Nothing, Float64} = nothing
    log::Bool = false
    warm_start::Bool = false
    initial_x::Union{Nothing, Vector{Float64}} = nothing
    initial_z::Union{Nothing, Vector{Float64}} = nothing
    required_assets::Vector{Int} = Int[]
    forbidden_assets::Vector{Int} = Int[]
    optimizer_attributes::Dict{String, Any} = Dict{String, Any}()
end

function _validate_options(options::SolverOptions)
    options.threads >= 0 ||
        throw(ArgumentError("threads must be nonnegative"))
    isfinite(options.tolerance) && options.tolerance > 0.0 ||
        throw(ArgumentError("tolerance must be positive and finite"))
    if options.time_limit !== nothing
        isfinite(options.time_limit) && options.time_limit > 0.0 ||
            throw(ArgumentError("time_limit must be positive and finite"))
    end
    return options
end

function _validate_initial_x(
    instance::MarkowitzInstance,
    options::SolverOptions,
)
    options.initial_x === nothing && return options
    initial_x = options.initial_x
    dimension = size(instance.B, 1)
    length(initial_x) == dimension ||
        throw(
            ArgumentError(
                "initial_x has length $(length(initial_x)); expected " *
                "$dimension",
            ),
        )
    all(isfinite, initial_x) ||
        throw(ArgumentError("initial_x contains a non-finite value"))
    tolerance = 1.0e-8
    minimum(initial_x) >= -tolerance ||
        throw(ArgumentError("initial_x violates the nonnegative domain"))
    maximum(initial_x) <= 1.0 + tolerance ||
        throw(ArgumentError("initial_x violates the unit-box domain"))
    sum(initial_x) <= instance.k + tolerance ||
        throw(
            ArgumentError(
                "initial_x violates the perspective budget sum(x) <= k",
            ),
        )
    return options
end

function _normalize_solver(solver::Union{Symbol, AbstractString})
    text = strip(String(solver))
    isempty(text) && throw(ArgumentError("solver must not be empty"))
    normalized = lowercase(text)
    if endswith(normalized, ".optimizer")
        package_alias = first(split(normalized, "."))
        if haskey(BUILTIN_SOLVERS, package_alias)
            normalized = package_alias
        end
    end
    if haskey(BUILTIN_SOLVERS, normalized)
        entry = BUILTIN_SOLVERS[normalized]
        label = normalized == "mosektools" ? "mosek" : normalized
        return OptimizerSpec(
            label,
            entry.package,
            entry.constructor,
            entry.profile,
            entry.conic,
        )
    end
    parts = split(text, ".")
    length(parts) in (1, 2) ||
        throw(
            ArgumentError(
                "custom solver must be Package or Package.Constructor",
            ),
        )
    all(part -> occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", part), parts) ||
        throw(ArgumentError("custom solver contains an invalid identifier"))
    package = Symbol(first(parts))
    constructor = Symbol(length(parts) == 1 ? "Optimizer" : last(parts))
    return OptimizerSpec(
        text,
        package,
        constructor,
        :generic,
        true,
    )
end

function _metadata_dictionary(path::AbstractString)
    isfile(path) || return Dict{String, Any}()
    return JSON3.read(read(path, String), Dict{String, Any})
end

function _required_metadata(metadata, key::String)
    haskey(metadata, key) ||
        throw(ArgumentError("bundle metadata is missing '$key'"))
    return metadata[key]
end

function _binary_file(root::AbstractString, files, key::String)
    haskey(files, key) ||
        throw(ArgumentError("bundle file table is missing '$key'"))
    path = abspath(joinpath(root, String(files[key])))
    relative_path = relpath(path, abspath(root))
    relative_parts = splitpath(relative_path)
    (isempty(relative_parts) || first(relative_parts) != "..") ||
        throw(ArgumentError("bundle file escapes its directory: $key"))
    isfile(path) ||
        throw(ArgumentError("bundle file does not exist: $path"))
    return path
end

function _mmap_vector(
    path::AbstractString,
    ::Type{T},
    length::Int,
) where {T}
    length >= 0 || throw(ArgumentError("negative binary array length"))
    expected_bytes = length * sizeof(T)
    actual_bytes = filesize(path)
    actual_bytes == expected_bytes ||
        throw(
            ArgumentError(
                "binary size mismatch for $(basename(path)): " *
                "expected $expected_bytes bytes, found $actual_bytes",
            ),
        )
    stream = open(path, "r")
    try
        return Mmap.mmap(stream, Vector{T}, (length,))
    finally
        close(stream)
    end
end

function _generator_metadata(metadata)
    result = Dict{String, Any}()
    generator = get(metadata, "generator", nothing)
    if generator !== nothing
        for (key, value) in pairs(generator)
            result[String(key)] = value
        end
    end
    return result
end

"""
Load the raw binary directory written by Python's
`save_instance_bundle`. Dense factor data is memory-mapped in Julia's
native column-major order, and the zero-based CSC indices are shifted
once without creating a dense constraint matrix.
"""
function _load_binary_bundle(
    root::AbstractString,
    metadata::Dict{String, Any},
)
    _required_metadata(metadata, "schema") ==
        "markowitz-perspective-bundle" ||
        throw(ArgumentError("not a Markowitz perspective bundle"))
    schema_version = Int(
        _required_metadata(metadata, "schema_version"),
    )
    schema_version == 1 ||
        throw(
            ArgumentError(
                "unsupported schema_version $schema_version; expected 1",
            ),
        )
    dimension = Int(_required_metadata(metadata, "dimension"))
    factors = Int(_required_metadata(metadata, "rank"))
    constraints = Int(_required_metadata(metadata, "constraint_rows"))
    nonzeros = Int(_required_metadata(metadata, "constraint_nnz"))
    dimension > 0 || throw(ArgumentError("dimension must be positive"))
    factors > 0 || throw(ArgumentError("rank must be positive"))
    constraints >= 0 ||
        throw(ArgumentError("constraint_rows must be nonnegative"))
    nonzeros >= 0 ||
        throw(ArgumentError("constraint_nnz must be nonnegative"))
    get(metadata, "matrix_order", "column-major") == "column-major" ||
        throw(ArgumentError("factor matrix must be column-major"))
    get(metadata, "constraint_storage", "csc-zero-based") ==
        "csc-zero-based" ||
        throw(ArgumentError("constraint matrix must use zero-based CSC"))

    files = _required_metadata(metadata, "files")
    B_values = _mmap_vector(
        _binary_file(root, files, "factor_loadings"),
        Float64,
        dimension * factors,
    )
    B = reshape(B_values, dimension, factors)
    mu = _mmap_vector(
        _binary_file(root, files, "expected_returns"),
        Float64,
        dimension,
    )
    lower = _mmap_vector(
        _binary_file(root, files, "lower_bounds"),
        Float64,
        constraints,
    )
    upper = _mmap_vector(
        _binary_file(root, files, "upper_bounds"),
        Float64,
        constraints,
    )
    anchor = _mmap_vector(
        _binary_file(root, files, "feasible_anchor"),
        Float64,
        dimension,
    )
    raw_colptr = _mmap_vector(
        _binary_file(root, files, "constraint_colptr"),
        Int64,
        dimension + 1,
    )
    raw_rowval = _mmap_vector(
        _binary_file(root, files, "constraint_rowval"),
        Int64,
        nonzeros,
    )
    nzval = _mmap_vector(
        _binary_file(root, files, "constraint_nzval"),
        Float64,
        nonzeros,
    )

    first(raw_colptr) == 0 ||
        throw(ArgumentError("constraint_colptr must start at zero"))
    last(raw_colptr) == nonzeros ||
        throw(ArgumentError("constraint_colptr has the wrong endpoint"))
    issorted(raw_colptr) ||
        throw(ArgumentError("constraint_colptr must be nondecreasing"))
    if !isempty(raw_rowval)
        minimum(raw_rowval) >= 0 &&
            maximum(raw_rowval) < constraints ||
            throw(ArgumentError("constraint_rowval contains an invalid row"))
    end
    all(isfinite, nzval) ||
        throw(ArgumentError("constraint_nzval contains a non-finite value"))

    colptr = Int.(raw_colptr) .+ 1
    rowval = Int.(raw_rowval) .+ 1
    C = SparseMatrixCSC{Float64, Int}(
        constraints,
        dimension,
        colptr,
        rowval,
        nzval,
    )
    k = Int(_required_metadata(metadata, "k"))
    perspective_weight = Float64(
        _required_metadata(metadata, "perspective_weight"),
    )
    return_reward = Float64(
        _required_metadata(metadata, "return_reward"),
    )

    metadata_copy = _generator_metadata(metadata)
    metadata_copy["schema_version"] = schema_version
    metadata_copy["bundle_source"] = abspath(root)
    metadata_copy["dimension"] = dimension
    metadata_copy["factors"] = factors
    metadata_copy["constraints"] = constraints
    metadata_copy["constraint_nnz"] = nonzeros

    return _validate_instance(
        MarkowitzInstance(
            B,
            mu,
            C,
            lower,
            upper,
            anchor,
            k,
            perspective_weight,
            return_reward,
            metadata_copy,
        ),
    )
end

function _validate_instance(instance::MarkowitzInstance)
    dimension, factors = size(instance.B)
    dimension > 0 || throw(ArgumentError("B must have at least one row"))
    factors > 0 || throw(ArgumentError("B must have at least one column"))
    all(isfinite, instance.B) ||
        throw(ArgumentError("B contains a non-finite value"))
    length(instance.mu) == dimension ||
        throw(ArgumentError("mu has the wrong length"))
    all(isfinite, instance.mu) ||
        throw(ArgumentError("mu contains a non-finite value"))
    size(instance.C, 2) == dimension ||
        throw(ArgumentError("C has the wrong number of columns"))
    constraints = size(instance.C, 1)
    length(instance.lower) == constraints ||
        throw(ArgumentError("lower has the wrong length"))
    length(instance.upper) == constraints ||
        throw(ArgumentError("upper has the wrong length"))
    all(value -> !isnan(value), instance.lower) ||
        throw(ArgumentError("lower contains NaN"))
    all(value -> !isnan(value), instance.upper) ||
        throw(ArgumentError("upper contains NaN"))
    all(instance.lower .<= instance.upper) ||
        throw(ArgumentError("a lower bound exceeds its upper bound"))
    1 <= instance.k <= dimension ||
        throw(ArgumentError("k must lie in {1, ..., d}"))
    isfinite(instance.perspective_weight) &&
        instance.perspective_weight > 0.0 ||
        throw(ArgumentError("perspective_weight must be positive"))
    isfinite(instance.return_reward) &&
        instance.return_reward >= 0.0 ||
        throw(ArgumentError("return_reward must be nonnegative"))
    if instance.anchor !== nothing
        length(instance.anchor) == dimension ||
            throw(ArgumentError("anchor has the wrong length"))
        all(isfinite, instance.anchor) ||
            throw(ArgumentError("anchor contains a non-finite value"))
    end
    return instance
end

"""
Load a cross-language binary instance bundle.

The directory contains `metadata.json`, raw little-endian Float64
arrays, and a zero-based CSC representation of `C`. The factor matrix
is stored column-major and is reshaped without a transpose. All budget
and return requirements are rows of `C`.
"""
function load_bundle(path::AbstractString)
    root = abspath(path)
    isdir(root) ||
        throw(ArgumentError("bundle path is not a directory: $root"))
    metadata_path = joinpath(root, "metadata.json")
    isfile(metadata_path) ||
        throw(ArgumentError("bundle is missing metadata.json"))
    metadata = _metadata_dictionary(metadata_path)
    return _load_binary_bundle(root, metadata)
end

function _lazy_import(package::Symbol)
    if !isdefined(@__MODULE__, package)
        Core.eval(
            @__MODULE__,
            Meta.parse("import $(String(package))"),
        )
    end
    # Resolve the newly imported binding in the latest world. This avoids
    # Julia 1.12's warning when a package is imported lazily from a function.
    return Core.eval(@__MODULE__, package)
end

function _load_optimizer(specification::OptimizerSpec)
    if specification.profile == :mosek
        # Loading MosekTools registers the MathOptInterface methods for
        # Mosek.Optimizer. It is intentionally imported only in this branch.
        _lazy_import(:MosekTools)
    end
    package = _lazy_import(specification.package)
    isdefined(package, specification.constructor) ||
        throw(
            ArgumentError(
                "$(specification.package) does not define " *
                "$(specification.constructor)",
            ),
        )
    return getfield(package, specification.constructor), package
end

function _apply_optimizer_attributes!(
    model::Model,
    attributes::Dict{String, Any},
)
    for name in sort!(collect(keys(attributes)))
        set_attribute(model, name, attributes[name])
    end
    return nothing
end

function _set_solver_options!(
    model::Model,
    specification::OptimizerSpec,
    package,
    options::SolverOptions,
)
    feasibility_tolerance =
        max(1.0e-9, min(1.0e-2, options.tolerance))

    if options.log
        unset_silent(model)
    else
        set_silent(model)
    end
    options.time_limit === nothing ||
        set_time_limit_sec(model, options.time_limit)

    if specification.profile == :gurobi
        set_attribute(model, "OutputFlag", options.log ? 1 : 0)
        set_attribute(model, "NonConvex", 0)
        set_attribute(model, "Method", 2)
        set_attribute(model, "Crossover", 0)
        set_attribute(model, "Threads", options.threads)
        set_attribute(model, "FeasibilityTol", feasibility_tolerance)
        set_attribute(model, "OptimalityTol", feasibility_tolerance)
        set_attribute(model, "BarConvTol", options.tolerance)
        set_attribute(model, "BarQCPConvTol", options.tolerance)
    elseif specification.profile == :mosek
        if options.threads > 0
            set_attribute(
                model,
                "MSK_IPAR_NUM_THREADS",
                options.threads,
            )
        end
        set_attribute(
            model,
            "MSK_IPAR_OPTIMIZER",
            getproperty(package, :MSK_OPTIMIZER_CONIC),
        )
        set_attribute(
            model,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
            options.tolerance,
        )
        set_attribute(
            model,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            feasibility_tolerance,
        )
        set_attribute(
            model,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            feasibility_tolerance,
        )
    elseif specification.profile == :clarabel
        set_attribute(model, "tol_gap_abs", options.tolerance)
        set_attribute(model, "tol_gap_rel", options.tolerance)
        set_attribute(model, "tol_feas", feasibility_tolerance)
    elseif specification.profile == :cosmo
        set_attribute(model, "eps_abs", feasibility_tolerance)
        set_attribute(model, "eps_rel", options.tolerance)
    end
    _apply_optimizer_attributes!(model, options.optimizer_attributes)
    return feasibility_tolerance
end

function _apply_warm_start!(
    instance::MarkowitzInstance,
    initial_x::Union{Nothing, Vector{Float64}},
    x,
    z,
    t,
    factor_exposure,
    risk_epigraph,
)
    source = initial_x === nothing ? instance.anchor : initial_x
    source_name = initial_x === nothing ? "feasible_anchor" : "initial_x"
    source === nothing && return false, "none"
    anchor_tolerance = 1.0e-8
    minimum(source) >= -anchor_tolerance || return false, source_name
    maximum(source) <= 1.0 + anchor_tolerance || return false, source_name
    sum(source) <= instance.k + anchor_tolerance ||
        return false, source_name
    anchor = clamp.(source, 0.0, 1.0)
    total = sum(anchor)
    total > 0.0 || return false, source_name
    scale = max(1.0, instance.k / total)
    z_start = min.(1.0, scale .* anchor)
    t_start = zeros(length(anchor))
    for i in eachindex(anchor)
        if z_start[i] > 0.0
            t_start[i] = anchor[i]^2 / z_start[i]
        end
    end
    factor_start = transpose(instance.B) * anchor

    set_start_value.(x, anchor)
    set_start_value.(z, z_start)
    set_start_value.(t, t_start)
    set_start_value.(factor_exposure, factor_start)
    set_start_value(risk_epigraph, sum(abs2, factor_start))
    return true, source_name
end

"""
Build the shared JuMP formulation used by every conic optimizer.

The rotated cone convention is

    (a, b, c) in RSOC  <=>  2 * a * b >= ||c||^2.

Thus `(risk_epigraph, 0.5, B' * x)` gives
`risk_epigraph >= ||B' * x||^2`, while
`(0.5 * t[i], z[i], x[i])` gives `t[i] * z[i] >= x[i]^2`.
The objective coefficients below therefore match the Python models.
"""
function _build_model(
    instance::MarkowitzInstance,
    specification::OptimizerSpec,
    optimizer,
    package,
    options::SolverOptions,
)
    model = Model(optimizer)
    set_string_names_on_creation(model, false)
    feasibility_tolerance = _set_solver_options!(
        model,
        specification,
        package,
        options,
    )

    dimension, factors = size(instance.B)
    @variable(model, 0.0 <= x[1:dimension] <= 1.0)
    @variable(model, 0.0 <= z[1:dimension] <= 1.0)
    @variable(model, t[1:dimension] >= 0.0)
    @variable(model, factor_exposure[1:factors])
    @variable(model, risk_epigraph >= 0.0)

    @constraint(model, sum(z) <= instance.k)
    for i in 1:dimension
        @constraint(model, x[i] <= z[i])
    end

    factor_expressions = transpose(instance.B) * x
    for factor in 1:factors
        @constraint(
            model,
            factor_exposure[factor] == factor_expressions[factor],
        )
    end
    @constraint(
        model,
        [risk_epigraph; 0.5; factor_exposure] in
        RotatedSecondOrderCone(),
    )
    for i in 1:dimension
        @constraint(
            model,
            [0.5 * t[i]; z[i]; x[i]] in
            RotatedSecondOrderCone(),
        )
    end

    if size(instance.C, 1) > 0
        constraint_expressions = instance.C * x
        for row in axes(instance.C, 1)
            lower = instance.lower[row]
            upper = instance.upper[row]
            if isfinite(lower) && isfinite(upper) &&
               abs(lower - upper) <=
               1.0e-14 * max(1.0, abs(lower), abs(upper))
                @constraint(
                    model,
                    constraint_expressions[row] == 0.5 * (lower + upper),
                )
            else
                if isfinite(lower)
                    @constraint(
                        model,
                        constraint_expressions[row] >= lower,
                    )
                end
                if isfinite(upper)
                    @constraint(
                        model,
                        constraint_expressions[row] <= upper,
                    )
                end
            end
        end
    end

    @objective(
        model,
        Min,
        0.5 * risk_epigraph +
        0.5 * instance.perspective_weight * sum(t) -
        instance.return_reward * dot(instance.mu, x),
    )

    # Gurobi and COSMO accept MOI primal starts. MOSEK's interior-point
    # optimizer and Clarabel do not use this conventional JuMP start.
    warm_start_supported =
        specification.profile in (:gurobi, :cosmo)
    warm_start_applied, warm_start_source = if options.warm_start &&
                                               warm_start_supported
        _apply_warm_start!(
            instance,
            options.initial_x,
            x,
            z,
            t,
            factor_exposure,
            risk_epigraph,
        )
    else
        false, "none"
    end

    variables = (
        x = x,
        z = z,
        t = t,
        factor_exposure = factor_exposure,
        risk_epigraph = risk_epigraph,
    )
    return (
        model,
        variables,
        feasibility_tolerance,
        warm_start_supported,
        warm_start_applied,
        warm_start_source,
    )
end

function _perspective_value(
    x::AbstractVector{<:Real},
    k::Int;
    tolerance::Float64,
)
    minimum(x) >= -tolerance || return Inf
    maximum(x) <= 1.0 + tolerance || return Inf
    sum(x) <= k + tolerance || return Inf
    positive = max.(Float64.(x), 0.0)
    values = positive[positive .> 0.0]
    isempty(values) && return 0.0
    if length(values) <= k
        return 0.5 * sum(abs2, values)
    end

    lower_scale = 1.0
    upper_scale = max(2.0, k / max(sum(values), eps(Float64)))
    while sum(min(1.0, upper_scale * value) for value in values) < k
        upper_scale *= 2.0
        isfinite(upper_scale) ||
            throw(ArgumentError("perspective evaluation failed to bracket"))
    end
    for _ in 1:80
        scale = 0.5 * (lower_scale + upper_scale)
        if sum(min(1.0, scale * value) for value in values) < k
            lower_scale = scale
        else
            upper_scale = scale
        end
    end
    result = 0.0
    for value in values
        z_value = max(value, min(1.0, upper_scale * value))
        result += value^2 / z_value
    end
    return 0.5 * result
end

function _external_objective(
    instance::MarkowitzInstance,
    x::Vector{Float64};
    tolerance::Float64,
)
    factor_exposure = transpose(instance.B) * x
    perspective = _perspective_value(
        x,
        instance.k;
        tolerance = tolerance,
    )
    objective =
        0.5 * sum(abs2, factor_exposure) +
        instance.perspective_weight * perspective -
        instance.return_reward * dot(instance.mu, x)
    return objective, perspective
end

function _diagnostics(
    instance::MarkowitzInstance,
    x::Vector{Float64},
)
    linear_violation = 0.0
    if size(instance.C, 1) > 0
        values = instance.C * x
        for row in eachindex(values)
            if isfinite(instance.lower[row])
                linear_violation = max(
                    linear_violation,
                    instance.lower[row] - values[row],
                )
            end
            if isfinite(instance.upper[row])
                linear_violation = max(
                    linear_violation,
                    values[row] - instance.upper[row],
                )
            end
        end
    end
    # The full-investment equality is a row of C, so it is already
    # included in linear_violation.
    budget_violation = 0.0
    box_violation = max(
        maximum((-value for value in x); init = 0.0),
        maximum((value - 1.0 for value in x); init = 0.0),
    )
    perspective_budget_violation = max(0.0, sum(x) - instance.k)
    perspective_domain_violation = max(
        0.0,
        box_violation,
        perspective_budget_violation,
    )
    violation = max(
        0.0,
        linear_violation,
        budget_violation,
        perspective_domain_violation,
    )
    return Dict{String, Any}(
        "linear_violation" => max(0.0, linear_violation),
        "budget_violation" => budget_violation,
        "box_violation" => max(0.0, box_violation),
        "perspective_budget_violation" =>
            perspective_budget_violation,
        "perspective_domain_violation" =>
            perspective_domain_violation,
        "violation" => violation,
    )
end

function _safe_query(query, default = nothing)
    try
        return query()
    catch
        return default
    end
end

function _package_version(package)
    return _safe_query(() -> string(Base.pkgversion(package)))
end

function _solver_snapshot(model, variables, fallback_solve_seconds)
    termination = termination_status(model)
    has_solution = has_values(model)
    return (
        termination = termination,
        primal = primal_status(model),
        dual = dual_status(model),
        raw = _safe_query(() -> raw_status(model), ""),
        has_solution = has_solution,
        x = has_solution ? Float64.(value.(variables.x)) : nothing,
        model_objective =
            has_solution ? Float64(objective_value(model)) : nothing,
        solver_time =
            _safe_query(() -> solve_time(model), fallback_solve_seconds),
        iterations = _safe_query(() -> barrier_iterations(model)),
        objective_bound = _safe_query(() -> objective_bound(model)),
        relative_gap = _safe_query(() -> relative_gap(model)),
    )
end

function _failure_kind(error)::String
    message = lowercase(sprint(showerror, error))
    if error isa MOI.UnsupportedConstraint ||
       occursin("unsupportedconstraint", message) ||
       occursin("rotatedsecondordercone", message) ||
       occursin("does not support the rotated", message)
        return "unsupported_formulation"
    elseif occursin("license", message) ||
       occursin("flexlm", message) ||
       occursin("grb_error_no_license", message) ||
       occursin("err_missing_license", message)
        return "license"
    elseif error isa LoadError ||
           error isa InitError ||
           occursin("package", message) &&
           occursin("not found", message)
        return "import"
    elseif error isa ArgumentError
        return "input"
    end
    return "solver"
end

function _error_result(
    solver,
    error;
    elapsed_seconds::Float64,
    phase::String,
)
    failure_kind = _failure_kind(error)
    unavailable = failure_kind in ("license", "import")
    unsupported = failure_kind == "unsupported_formulation"
    return Dict{String, Any}(
        "solver" => String(solver),
        "language" => "julia",
        "status" => unsupported ?
            "unsupported" : unavailable ? "unavailable" : "error",
        "success" => false,
        "has_solution" => false,
        "x" => nothing,
        "failure_kind" => failure_kind,
        "phase" => phase,
        "message" => sprint(showerror, error),
        "total_seconds" => elapsed_seconds,
        "julia_version" => string(VERSION),
    )
end

"""
Solve an already-loaded instance with a JuMP optimizer.

Built-in aliases are Gurobi, MosekTools, Clarabel, COSMO, and HiGHS.
Custom `Package` or `Package.Constructor` names are imported lazily.
The exact model is conic, so HiGHS is reported as unsupported instead
of silently replacing the perspective relaxation by an approximation.
"""
function solve_instance(
    instance::MarkowitzInstance;
    solver::Union{Symbol, AbstractString},
    options::SolverOptions = SolverOptions(),
    load_seconds::Float64 = 0.0,
)
    total_start = time_ns()
    specification = try
        _normalize_solver(solver)
    catch error
        return _error_result(
            solver,
            error;
            elapsed_seconds = load_seconds +
                (time_ns() - total_start) / 1.0e9,
            phase = "configuration",
        )
    end
    try
        _validate_options(options)
        _validate_instance(instance)
        _validate_initial_x(instance, options)
    catch error
        return _error_result(
            specification.label,
            error;
            elapsed_seconds = load_seconds +
                (time_ns() - total_start) / 1.0e9,
            phase = "configuration",
        )
    end

    if !specification.conic
        return _error_result(
            specification.label,
            ArgumentError(
                "$(specification.label) does not support the rotated " *
                "second-order cones required by the exact perspective " *
                "formulation",
            );
            elapsed_seconds = load_seconds +
                (time_ns() - total_start) / 1.0e9,
            phase = "formulation_support",
        )
    end

    backend_load_start = time_ns()
    optimizer, package = try
        _load_optimizer(specification)
    catch error
        return _error_result(
            specification.label,
            error;
            elapsed_seconds = load_seconds +
                (time_ns() - total_start) / 1.0e9,
            phase = "backend_import",
        )
    end
    backend_load_seconds = (time_ns() - backend_load_start) / 1.0e9

    build_start = time_ns()
    model, variables, feasibility_tolerance, warm_start_supported,
        warm_start_applied, warm_start_source = try
        # The solver package was imported lazily moments ago. invokelatest
        # makes its newly registered MOI methods visible on this first call.
        Base.invokelatest(
            _build_model,
            instance,
            specification,
            optimizer,
            package,
            options,
        )
    catch error
        return _error_result(
            specification.label,
            error;
            elapsed_seconds = load_seconds +
                (time_ns() - total_start) / 1.0e9,
            phase = "model_build",
        )
    end
    build_seconds = (time_ns() - build_start) / 1.0e9

    solve_start = time_ns()
    try
        Base.invokelatest(optimize!, model)
    catch error
        return _error_result(
            specification.label,
            error;
            elapsed_seconds = load_seconds +
                (time_ns() - total_start) / 1.0e9,
            phase = "optimize",
        )
    end
    solve_seconds = (time_ns() - solve_start) / 1.0e9

    # Attribute methods are registered by the lazily imported solver package.
    snapshot = Base.invokelatest(
        _solver_snapshot,
        model,
        variables,
        solve_seconds,
    )
    termination = snapshot.termination
    primal = snapshot.primal
    dual = snapshot.dual
    raw = snapshot.raw
    has_solution = snapshot.has_solution
    status = if termination == MOI.OPTIMAL && has_solution
        "optimal"
    elseif termination == MOI.TIME_LIMIT
        "time_limit"
    elseif termination == MOI.ITERATION_LIMIT
        "iteration_limit"
    elseif has_solution
        "suboptimal"
    elseif termination == MOI.INFEASIBLE
        "infeasible"
    elseif termination == MOI.DUAL_INFEASIBLE
        "dual_infeasible"
    elseif termination == MOI.INFEASIBLE_OR_UNBOUNDED
        "infeasible_or_unbounded"
    else
        "not_optimal"
    end

    postprocess_start = time_ns()
    x_value = snapshot.x
    model_objective = snapshot.model_objective
    external_objective = nothing
    perspective_value = nothing
    diagnostics = Dict{String, Any}(
        "linear_violation" => nothing,
        "budget_violation" => nothing,
        "box_violation" => nothing,
        "perspective_budget_violation" => nothing,
        "perspective_domain_violation" => nothing,
        "violation" => nothing,
    )
    if x_value !== nothing
        external_objective, perspective_value = _external_objective(
            instance,
            x_value;
            tolerance = feasibility_tolerance,
        )
        diagnostics = _diagnostics(instance, x_value)
    end
    postprocess_seconds = (time_ns() - postprocess_start) / 1.0e9

    solver_time = snapshot.solver_time
    iterations = snapshot.iterations
    result = Dict{String, Any}(
        "solver" => specification.label,
        "optimizer_package" => String(specification.package),
        "optimizer_constructor" => String(specification.constructor),
        "optimizer_profile" => String(specification.profile),
        "formulation" => "rotated_second_order_cone",
        "language" => "julia",
        "status" => status,
        "success" => status == "optimal",
        "has_solution" => has_solution,
        "x" => x_value,
        "model_objective" => model_objective,
        "external_objective" => external_objective,
        "perspective_value" => perspective_value,
        "objective_scaling_gap" =>
            model_objective === nothing ||
            external_objective === nothing ?
            nothing : model_objective - external_objective,
        "termination_status" => string(termination),
        "primal_status" => string(primal),
        "dual_status" => string(dual),
        "raw_status" => string(raw),
        "message" => string(raw),
        "dimension" => size(instance.B, 1),
        "factors" => size(instance.B, 2),
        "constraints" => size(instance.C, 1),
        "iterations" => iterations,
        "load_seconds" => load_seconds,
        "backend_load_seconds" => backend_load_seconds,
        "build_seconds" => build_seconds,
        "solve_seconds" => solve_seconds,
        "solver_solve_seconds" => solver_time,
        "effective_tolerance" => feasibility_tolerance,
        "solver_tolerance_profile_applied" =>
            specification.profile != :generic,
        "postprocess_seconds" => postprocess_seconds,
        "total_seconds" => load_seconds +
            (time_ns() - total_start) / 1.0e9,
        "warm_start_requested" => options.warm_start,
        "initial_x_supplied" => options.initial_x !== nothing,
        "warm_start_supported" => warm_start_supported,
        "warm_start_applied" => warm_start_applied,
        "warm_start_source" => warm_start_source,
        "solver_package_version" => _package_version(package),
        "julia_version" => string(VERSION),
        "objective_scaling" => Dict{String, Any}(
            "risk_epigraph_coefficient" => 0.5,
            "perspective_epigraph_coefficient" =>
                0.5 * instance.perspective_weight,
            "return_coefficient" => -instance.return_reward,
            "risk_cone" =>
                "(s, 0.5, B' x): s >= ||B' x||^2",
            "perspective_cones" =>
                "(0.5 t_i, z_i, x_i): t_i z_i >= x_i^2",
        ),
        "solver_details" => Dict{String, Any}(
            "threads" => options.threads,
            "requested_tolerance" => options.tolerance,
            "feasibility_tolerance" => feasibility_tolerance,
            "time_limit" => options.time_limit,
            "log" => options.log,
            "optimizer_attributes" =>
                copy(options.optimizer_attributes),
            "objective_bound" => snapshot.objective_bound,
            "relative_gap" => snapshot.relative_gap,
        ),
    )
    merge!(result, diagnostics)
    return result
end

function _mapping_get(options, key::String, default)
    options === nothing && return default
    for candidate in (key, Symbol(key))
        try
            return options[candidate]
        catch
        end
    end
    try
        return getproperty(options, Symbol(key))
    catch
        return default
    end
end

function _coerce_bool(value, name::String)
    value isa Bool && return value
    text = lowercase(string(value))
    text in ("true", "1") && return true
    text in ("false", "0") && return false
    throw(ArgumentError("$name must be Boolean"))
end

function _coerce_optimizer_attributes(value)
    value === nothing && return Dict{String, Any}()
    attributes = Dict{String, Any}()
    try
        for (key, item) in pairs(value)
            attributes[String(key)] = item
        end
    catch error
        throw(
            ArgumentError(
                "optimizer_attributes must be a mapping: " *
                sprint(showerror, error),
            ),
        )
    end
    return attributes
end

function _coerce_initial_x(value)
    value === nothing && return nothing
    try
        return Float64.(collect(value))
    catch error
        throw(
            ArgumentError(
                "initial_x must be a one-dimensional numeric sequence: " *
                sprint(showerror, error),
            ),
        )
    end
end

function _coerce_options(options)
    options isa SolverOptions && return options
    raw_time_limit = _mapping_get(options, "time_limit", nothing)
    time_limit = if raw_time_limit === nothing ||
                    lowercase(string(raw_time_limit)) == "none"
        nothing
    else
        Float64(raw_time_limit)
    end
    return SolverOptions(
        threads = Int(_mapping_get(options, "threads", 0)),
        tolerance = Float64(
            _mapping_get(options, "tolerance", 1.0e-6),
        ),
        time_limit = time_limit,
        log = _coerce_bool(
            _mapping_get(options, "log", false),
            "log",
        ),
        warm_start = _coerce_bool(
            _mapping_get(options, "warm_start", true),
            "warm_start",
        ),
        initial_x = _coerce_initial_x(
            _mapping_get(options, "initial_x", nothing),
        ),
        optimizer_attributes = _coerce_optimizer_attributes(
            _mapping_get(options, "optimizer_attributes", nothing),
        ),
    )
end

function solve_bundle(
    path::AbstractString;
    solver::Union{Symbol, AbstractString},
    options::SolverOptions = SolverOptions(),
)
    load_start = time_ns()
    instance = try
        load_bundle(path)
    catch error
        return _error_result(
            solver,
            error;
            elapsed_seconds = (time_ns() - load_start) / 1.0e9,
            phase = "input",
        )
    end
    load_seconds = (time_ns() - load_start) / 1.0e9
    return solve_instance(
        instance;
        solver = solver,
        options = options,
        load_seconds = load_seconds,
    )
end

function solve_bundle(
    path::AbstractString,
    solver::Union{Symbol, AbstractString},
    options = nothing,
)
    return solve_bundle(
        path;
        solver = solver,
        options = _coerce_options(options),
    )
end

function _validate_binary_options(
    instance::MarkowitzInstance,
    options::BinarySolverOptions,
)
    options.threads >= 0 ||
        throw(ArgumentError("threads must be nonnegative"))
    isfinite(options.tolerance) && options.tolerance > 0.0 ||
        throw(ArgumentError("tolerance must be positive and finite"))
    isfinite(options.relative_gap) && options.relative_gap >= 0.0 ||
        throw(ArgumentError("relative_gap must be finite and nonnegative"))
    isfinite(options.absolute_gap) && options.absolute_gap >= 0.0 ||
        throw(ArgumentError("absolute_gap must be finite and nonnegative"))
    if options.time_limit !== nothing
        isfinite(options.time_limit) && options.time_limit > 0.0 ||
            throw(ArgumentError("time_limit must be positive and finite"))
    end

    dimension = size(instance.B, 1)
    required = options.required_assets
    forbidden = options.forbidden_assets
    for (name, values) in (
        ("required_assets", required),
        ("forbidden_assets", forbidden),
    )
        length(unique(values)) == length(values) ||
            throw(ArgumentError("$name contains duplicates"))
        all(index -> 1 <= index <= dimension, values) ||
            throw(ArgumentError("$name contains an out-of-range index"))
    end
    isempty(intersect(required, forbidden)) ||
        throw(ArgumentError("required and forbidden assets overlap"))
    length(required) <= instance.k ||
        throw(ArgumentError("required assets exceed the cardinality budget"))

    if options.initial_x !== nothing
        length(options.initial_x) == dimension ||
            throw(ArgumentError("initial_x has the wrong dimension"))
        all(isfinite, options.initial_x) ||
            throw(ArgumentError("initial_x contains a non-finite value"))
    end
    if options.initial_z !== nothing
        length(options.initial_z) == dimension ||
            throw(ArgumentError("initial_z has the wrong dimension"))
        all(isfinite, options.initial_z) ||
            throw(ArgumentError("initial_z contains a non-finite value"))
    end
    return options
end

function _set_binary_solver_options!(
    model::Model,
    specification::OptimizerSpec,
    package,
    options::BinarySolverOptions,
)
    feasibility_tolerance =
        max(1.0e-9, min(1.0e-2, options.tolerance))
    options.log ? unset_silent(model) : set_silent(model)
    options.time_limit === nothing ||
        set_time_limit_sec(model, options.time_limit)

    if specification.profile == :gurobi
        set_attribute(model, "OutputFlag", options.log ? 1 : 0)
        set_attribute(model, "NonConvex", 0)
        set_attribute(model, "Threads", options.threads)
        set_attribute(model, "MIPFocus", 0)
        set_attribute(model, "MIPGap", options.relative_gap)
        set_attribute(model, "MIPGapAbs", options.absolute_gap)
        set_attribute(model, "FeasibilityTol", feasibility_tolerance)
        set_attribute(model, "OptimalityTol", feasibility_tolerance)
        set_attribute(model, "BarQCPConvTol", options.tolerance)
    elseif specification.profile == :mosek
        if options.threads > 0
            set_attribute(
                model,
                "MSK_IPAR_NUM_THREADS",
                options.threads,
            )
        end
        set_attribute(
            model,
            "MSK_DPAR_MIO_TOL_REL_GAP",
            options.relative_gap,
        )
        set_attribute(
            model,
            "MSK_DPAR_MIO_TOL_ABS_GAP",
            options.absolute_gap,
        )
        set_attribute(
            model,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
            options.tolerance,
        )
        set_attribute(
            model,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            feasibility_tolerance,
        )
        set_attribute(
            model,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            feasibility_tolerance,
        )
    end
    _apply_optimizer_attributes!(model, options.optimizer_attributes)
    return feasibility_tolerance
end

function _apply_binary_warm_start!(
    instance::MarkowitzInstance,
    options::BinarySolverOptions,
    x,
    z,
    t,
    factor_exposure,
    risk_epigraph,
)
    options.warm_start || return false
    options.initial_x === nothing && return false
    x_start = clamp.(options.initial_x, 0.0, 1.0)
    z_start = if options.initial_z === nothing
        Float64.(abs.(x_start) .> 1.0e-9)
    else
        Float64.(clamp.(round.(options.initial_z), 0.0, 1.0))
    end
    z_start[options.required_assets] .= 1.0
    z_start[options.forbidden_assets] .= 0.0
    sum(z_start) <= instance.k || return false
    maximum(x_start .- z_start) <= 1.0e-7 || return false
    t_start = zeros(length(x_start))
    for index in eachindex(x_start)
        if z_start[index] > 0.5
            t_start[index] = x_start[index]^2
        end
    end
    factor_start = transpose(instance.B) * x_start
    set_start_value.(x, x_start)
    set_start_value.(z, z_start)
    set_start_value.(t, t_start)
    set_start_value.(factor_exposure, factor_start)
    set_start_value(risk_epigraph, sum(abs2, factor_start))
    return true
end

function _build_binary_model(
    instance::MarkowitzInstance,
    specification::OptimizerSpec,
    optimizer,
    package,
    options::BinarySolverOptions,
)
    model = Model(optimizer)
    set_string_names_on_creation(model, false)
    feasibility_tolerance = _set_binary_solver_options!(
        model,
        specification,
        package,
        options,
    )
    dimension, factors = size(instance.B)
    @variable(model, 0.0 <= x[1:dimension] <= 1.0)
    @variable(model, z[1:dimension], Bin)
    @variable(model, t[1:dimension] >= 0.0)
    @variable(model, factor_exposure[1:factors])
    @variable(model, risk_epigraph >= 0.0)

    @constraint(model, sum(z) <= instance.k)
    @constraint(model, [index = 1:dimension], x[index] <= z[index])
    for index in options.required_assets
        @constraint(model, z[index] == 1.0)
    end
    for index in options.forbidden_assets
        @constraint(model, z[index] == 0.0)
    end
    factor_expressions = transpose(instance.B) * x
    @constraint(
        model,
        [factor = 1:factors],
        factor_exposure[factor] == factor_expressions[factor],
    )
    @constraint(
        model,
        [risk_epigraph; 0.5; factor_exposure] in
        RotatedSecondOrderCone(),
    )
    @constraint(
        model,
        [index = 1:dimension],
        [0.5 * t[index]; z[index]; x[index]] in
        RotatedSecondOrderCone(),
    )

    if size(instance.C, 1) > 0
        values = instance.C * x
        for row in axes(instance.C, 1)
            lower = instance.lower[row]
            upper = instance.upper[row]
            if isfinite(lower) && isfinite(upper) &&
               abs(lower - upper) <=
               1.0e-14 * max(1.0, abs(lower), abs(upper))
                @constraint(model, values[row] == 0.5 * (lower + upper))
            else
                if isfinite(lower)
                    @constraint(model, values[row] >= lower)
                end
                if isfinite(upper)
                    @constraint(model, values[row] <= upper)
                end
            end
        end
    end

    @objective(
        model,
        Min,
        0.5 * risk_epigraph +
        0.5 * instance.perspective_weight * sum(t) -
        instance.return_reward * dot(instance.mu, x),
    )
    warm_start_applied = _apply_binary_warm_start!(
        instance,
        options,
        x,
        z,
        t,
        factor_exposure,
        risk_epigraph,
    )
    variables = (
        x = x,
        z = z,
        t = t,
        factor_exposure = factor_exposure,
        risk_epigraph = risk_epigraph,
    )
    return model, variables, feasibility_tolerance, warm_start_applied
end

function _binary_solver_snapshot(model, variables, fallback_solve_seconds)
    termination = termination_status(model)
    has_solution = has_values(model)
    return (
        termination = termination,
        primal = primal_status(model),
        raw = _safe_query(() -> raw_status(model), ""),
        has_solution = has_solution,
        x = has_solution ? Float64.(value.(variables.x)) : nothing,
        z = has_solution ? Float64.(value.(variables.z)) : nothing,
        model_objective =
            has_solution ? Float64(objective_value(model)) : nothing,
        solver_time =
            _safe_query(() -> solve_time(model), fallback_solve_seconds),
        objective_bound = _safe_query(() -> objective_bound(model)),
        relative_gap = _safe_query(() -> relative_gap(model)),
        node_count = _safe_query(() -> node_count(model)),
        solution_count = _safe_query(() -> result_count(model)),
    )
end

function _binary_external_objective(
    instance::MarkowitzInstance,
    x::Vector{Float64},
)
    exposure = transpose(instance.B) * x
    return 0.5 * sum(abs2, exposure) +
           0.5 * instance.perspective_weight * sum(abs2, x) -
           instance.return_reward * dot(instance.mu, x)
end

function solve_binary_instance(
    instance::MarkowitzInstance;
    solver::Union{Symbol, AbstractString},
    options::BinarySolverOptions = BinarySolverOptions(),
    load_seconds::Float64 = 0.0,
)
    total_start = time_ns()
    specification = try
        _normalize_solver(solver)
    catch error
        return _error_result(
            solver,
            error;
            elapsed_seconds = load_seconds +
                (time_ns() - total_start) / 1.0e9,
            phase = "configuration",
        )
    end
    try
        _validate_instance(instance)
        _validate_binary_options(instance, options)
    catch error
        return _error_result(
            specification.label,
            error;
            elapsed_seconds = load_seconds +
                (time_ns() - total_start) / 1.0e9,
            phase = "configuration",
        )
    end
    if !specification.conic
        return _error_result(
            specification.label,
            ArgumentError(
                "$(specification.label) does not support the mixed-integer " *
                "rotated cones required by the perspective formulation",
            );
            elapsed_seconds = load_seconds +
                (time_ns() - total_start) / 1.0e9,
            phase = "formulation_support",
        )
    end

    backend_start = time_ns()
    optimizer, package = try
        _load_optimizer(specification)
    catch error
        return _error_result(
            specification.label,
            error;
            elapsed_seconds = load_seconds +
                (time_ns() - total_start) / 1.0e9,
            phase = "backend_import",
        )
    end
    backend_load_seconds = (time_ns() - backend_start) / 1.0e9
    build_start = time_ns()
    model, variables, feasibility_tolerance, warm_start_applied = try
        Base.invokelatest(
            _build_binary_model,
            instance,
            specification,
            optimizer,
            package,
            options,
        )
    catch error
        return _error_result(
            specification.label,
            error;
            elapsed_seconds = load_seconds +
                (time_ns() - total_start) / 1.0e9,
            phase = "model_build",
        )
    end
    build_seconds = (time_ns() - build_start) / 1.0e9
    solve_start = time_ns()
    try
        Base.invokelatest(optimize!, model)
    catch error
        return _error_result(
            specification.label,
            error;
            elapsed_seconds = load_seconds +
                (time_ns() - total_start) / 1.0e9,
            phase = "optimize",
        )
    end
    solve_seconds = (time_ns() - solve_start) / 1.0e9
    snapshot = Base.invokelatest(
        _binary_solver_snapshot,
        model,
        variables,
        solve_seconds,
    )
    termination = snapshot.termination
    has_solution = snapshot.has_solution
    status = if termination == MOI.OPTIMAL && has_solution
        "optimal"
    elseif termination == MOI.TIME_LIMIT
        "time_limit"
    elseif termination == MOI.ITERATION_LIMIT
        "iteration_limit"
    elseif has_solution
        "suboptimal"
    elseif termination == MOI.INFEASIBLE
        "infeasible"
    elseif termination == MOI.INFEASIBLE_OR_UNBOUNDED
        "infeasible_or_unbounded"
    else
        "not_optimal"
    end
    x_value = snapshot.x
    external_objective = x_value === nothing ? nothing :
        _binary_external_objective(instance, x_value)
    diagnostics = x_value === nothing ? Dict{String, Any}() :
        _diagnostics(instance, x_value)
    result = Dict{String, Any}(
        "solver" => specification.label,
        "optimizer_package" => String(specification.package),
        "optimizer_constructor" => String(specification.constructor),
        "optimizer_profile" => String(specification.profile),
        "language" => "julia",
        "formulation" => "exact_binary_perspective_misocp",
        "status" => status,
        "success" => status == "optimal",
        "has_solution" => has_solution,
        "x" => x_value,
        "selectors" => snapshot.z,
        "model_objective" => snapshot.model_objective,
        "external_objective" => external_objective,
        "termination_status" => string(termination),
        "primal_status" => string(snapshot.primal),
        "raw_status" => string(snapshot.raw),
        "message" => string(snapshot.raw),
        "load_seconds" => load_seconds,
        "backend_load_seconds" => backend_load_seconds,
        "build_seconds" => build_seconds,
        "solve_seconds" => solve_seconds,
        "solver_solve_seconds" => snapshot.solver_time,
        "total_seconds" => load_seconds +
            (time_ns() - total_start) / 1.0e9,
        "warm_start_requested" => options.warm_start,
        "warm_start_applied" => warm_start_applied,
        "node_count" => snapshot.node_count,
        "solution_count" => snapshot.solution_count,
        "solver_objective_bound" => snapshot.objective_bound,
        "solver_mip_gap" => snapshot.relative_gap,
        "effective_tolerance" => feasibility_tolerance,
        "solver_package_version" => _package_version(package),
        "julia_version" => string(VERSION),
        "solver_details" => Dict{String, Any}(
            "threads" => options.threads,
            "requested_tolerance" => options.tolerance,
            "relative_gap" => options.relative_gap,
            "absolute_gap" => options.absolute_gap,
            "time_limit" => options.time_limit,
            "objective_bound" => snapshot.objective_bound,
            "optimizer_attributes" =>
                copy(options.optimizer_attributes),
        ),
    )
    merge!(result, diagnostics)
    return result
end

function _coerce_binary_options(
    options,
    required_assets::Vector{Int},
    forbidden_assets::Vector{Int},
)
    options isa BinarySolverOptions && return options
    raw_time_limit = _mapping_get(options, "time_limit", nothing)
    time_limit = if raw_time_limit === nothing ||
                    lowercase(string(raw_time_limit)) == "none"
        nothing
    else
        Float64(raw_time_limit)
    end
    return BinarySolverOptions(
        threads = Int(_mapping_get(options, "threads", 0)),
        tolerance = Float64(
            _mapping_get(options, "tolerance", 1.0e-8),
        ),
        relative_gap = Float64(
            _mapping_get(options, "relative_gap", 1.0e-4),
        ),
        absolute_gap = Float64(
            _mapping_get(options, "absolute_gap", 0.0),
        ),
        time_limit = time_limit,
        log = _coerce_bool(_mapping_get(options, "log", false), "log"),
        warm_start = _coerce_bool(
            _mapping_get(options, "warm_start", false),
            "warm_start",
        ),
        initial_x = _coerce_initial_x(
            _mapping_get(options, "initial_x", nothing),
        ),
        initial_z = _coerce_initial_x(
            _mapping_get(options, "initial_z", nothing),
        ),
        required_assets = required_assets,
        forbidden_assets = forbidden_assets,
        optimizer_attributes = _coerce_optimizer_attributes(
            _mapping_get(options, "optimizer_attributes", nothing),
        ),
    )
end

function solve_binary_bundle(
    path::AbstractString,
    solver::Union{Symbol, AbstractString},
    options = nothing,
    required_assets = Int[],
    forbidden_assets = Int[],
)
    load_start = time_ns()
    instance = try
        load_bundle(path)
    catch error
        return _error_result(
            solver,
            error;
            elapsed_seconds = (time_ns() - load_start) / 1.0e9,
            phase = "input",
        )
    end
    load_seconds = (time_ns() - load_start) / 1.0e9
    required = Int[Int(value) + 1 for value in required_assets]
    forbidden = Int[Int(value) + 1 for value in forbidden_assets]
    binary_options = try
        _coerce_binary_options(options, required, forbidden)
    catch error
        return _error_result(
            solver,
            error;
            elapsed_seconds = (time_ns() - load_start) / 1.0e9,
            phase = "configuration",
        )
    end
    return solve_binary_instance(
        instance;
        solver = solver,
        options = binary_options,
        load_seconds = load_seconds,
    )
end

function _json_safe(value)
    if value === nothing ||
       value isa String ||
       value isa Bool ||
       value isa Integer
        return value
    elseif value isa AbstractFloat
        return isfinite(value) ? value : nothing
    elseif value isa Symbol
        return String(value)
    elseif value isa AbstractDict
        return Dict(
            String(key) => _json_safe(item)
            for (key, item) in pairs(value)
        )
    elseif value isa Tuple || value isa AbstractArray
        return [_json_safe(item) for item in value]
    end
    return string(value)
end

"""
Write a strict-JSON result. Large solution vectors are written to a
neighboring raw Float64 sidecar to avoid decimal JSON serialization.
"""
function write_result(
    result::AbstractDict,
    path::AbstractString;
    inline_solution_limit::Int = 10_000,
)
    inline_solution_limit >= 0 ||
        throw(ArgumentError("inline_solution_limit must be nonnegative"))
    output_path = abspath(path)
    mkpath(dirname(output_path))
    output = Dict{String, Any}(
        String(key) => value for (key, value) in pairs(result)
    )
    x_value = get(output, "x", nothing)
    if x_value isa AbstractVector &&
       length(x_value) > inline_solution_limit
        stem = splitext(basename(output_path))[1]
        sidecar_name = "$(stem)_x.f64"
        sidecar_path = joinpath(dirname(output_path), sidecar_name)
        temporary_sidecar = sidecar_path * ".tmp"
        open(temporary_sidecar, "w") do stream
            write(stream, Float64.(x_value))
        end
        mv(temporary_sidecar, sidecar_path; force = true)
        delete!(output, "x")
        output["x_file"] = sidecar_name
        output["x_format"] = "little-endian-float64"
        output["x_length"] = length(x_value)
    end

    temporary_json = output_path * ".tmp"
    open(temporary_json, "w") do stream
        JSON3.write(stream, _json_safe(output))
    end
    mv(temporary_json, output_path; force = true)
    return output_path
end

function _usage()
    return """
Usage:
  julia --project=. MarkowitzCommercial.jl \\
      --solver gurobi|mosek --bundle INSTANCE_DIRECTORY --result RESULT.json \\
      [--threads N] [--tolerance EPS] [--time-limit SECONDS] \\
      [--log] [--no-warm-start] [--inline-solution-limit N]
"""
end

function _parse_cli(args)
    values = Dict{String, String}()
    flags = Set{String}()
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument in ("--log", "--no-warm-start", "--help", "-h")
            push!(flags, argument)
            index += 1
            continue
        end
        startswith(argument, "--") ||
            throw(ArgumentError("unexpected argument: $argument"))
        index == length(args) &&
            throw(ArgumentError("missing value after $argument"))
        values[argument] = args[index + 1]
        index += 2
    end
    return values, flags
end

function main(args = ARGS)
    values, flags = try
        _parse_cli(args)
    catch error
        println(stderr, sprint(showerror, error))
        println(stderr, _usage())
        return 2
    end
    if "--help" in flags || "-h" in flags
        println(_usage())
        return 0
    end
    for required in ("--solver", "--bundle", "--result")
        if !haskey(values, required)
            println(stderr, "missing required argument: $required")
            println(stderr, _usage())
            return 2
        end
    end

    options = try
        SolverOptions(
            threads = parse(Int, get(values, "--threads", "0")),
            tolerance = parse(
                Float64,
                get(values, "--tolerance", "1e-6"),
            ),
            time_limit = haskey(values, "--time-limit") ?
                parse(Float64, values["--time-limit"]) : nothing,
            log = "--log" in flags,
            warm_start = !("--no-warm-start" in flags),
        )
    catch error
        println(stderr, sprint(showerror, error))
        return 2
    end
    inline_limit = try
        parse(
            Int,
            get(values, "--inline-solution-limit", "10000"),
        )
    catch error
        println(stderr, sprint(showerror, error))
        return 2
    end

    result = solve_bundle(
        values["--bundle"];
        solver = values["--solver"],
        options = options,
    )
    try
        write_result(
            result,
            values["--result"];
            inline_solution_limit = inline_limit,
        )
    catch error
        println(stderr, sprint(showerror, error))
        return 1
    end
    return get(result, "status", "error") in
        ("optimal", "suboptimal", "unavailable") ? 0 : 1
end

end # module MarkowitzCommercial

if abspath(PROGRAM_FILE) == @__FILE__
    exit(MarkowitzCommercial.main())
end
