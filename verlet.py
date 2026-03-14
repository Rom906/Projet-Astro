def verlet(vector_list, differential_equation, t, h, number_of_steps):
    """
    Verlet single step compatible with `compute_solution`.

    Expected inputs:
      - vector_list: List[Vector] where vector_list[0] is the most recent state Y_n.
      - differential_equation: function f(t, Y) -> Vector representing Y'.
      - t: current time (float)
      - h: timestep (float)
      - number_of_steps: not used for single-step RK4 but kept for API compatibility.

    Returns:
      - Vector: the estimated Y_{n+1}
    """
    result = vector_list
    for i in range(number_of_steps):
        an = result[-1][0]
        vn = result[-1][1]
        fn = differential_equation([an, vn])
        result.append([an + h * vn + (h**2)/2 * fn, vn + h/2 * (fn + differential_equation(fn))])
