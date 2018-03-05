# To solve the cubic, perform binary search.

# Solves (finds a single zero of) cubic equations of the form x^3 + ax^2 +bx + c
# by bisection.
function solve_bisection(f, x_min, x_max, x_tol=1e-8, y_tol=1e-6)
    s_min = f(x_min)
    s_max = f(x_max)

    if s_min > 0
        throw("s_min must be <= 0!")
    end
    if s_max < 0
        throw("s_max must be >= 0!")
    end
    if x_min > x_max
        throw("You must have x_min be larger than x_max")
    end

    # Check the endpoints first!
    if abs(s_min) <= y_tol
        return s_min
    elseif abs(s_max) <= y_tol
        return s_max
    end

    while (x_max - x_min > x_tol)
        x_med = (x_max + x_min)/2
        s_med = f(x_med)
        if abs(s_med) <= y_tol
            return x_med
        end
        if s_med > 0
            s_max, x_max = s_med,  x_med
        else
            s_min, x_min = s_med, x_med
        end
    end

    x_med = (x_max + x_min)/2

    if f(x_med) > y_tol
        warn("No zero found in the interval with tolerance < $(y_tol), possibly returning garbage.")
    end

    return x_med
end