import copy

'''
    Pure Python/NumPy implementation of the Nelder-Mead algorithm.
    Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
'''


def nelder_mead(f, x_start,
                step=0.1, improvement_threshold=10e-6,
                max_iterations_since_improvement=10, max_iterations=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)
        return: tuple (best parameter array, best score)
    '''

    # initialization
    dimensions = len(x_start)
    previous_best = f(x_start)
    iterations_since_improvement = 0
    record = [[x_start, previous_best]]

    for i in range(dimensions):
        x = copy.copy(x_start)
        x[i] += step
        score = f(x)
        record.append([x, score])

    # simplex iteration
    iterations = 0
    while(True):
        # order the record of [x, score] pairs by score
        record.sort(key=lambda x: x[1])
        best = record[0][1]
        
        print('...best so far:', best)
        
        # return the least-scoring pair after too many iterations total
        if max_iterations and iterations >= max_iterations: return record[0]
        iterations += 1
        
        # check if the score has improved (i.e., decreased)
        if best < previous_best - improvement_threshold:
            iterations_since_improvement = 0
            previous_best = best
        else: iterations_since_improvement += 1
        
        # return the least-scoring pair after too many iterations without improvement
        if iterations_since_improvement >= max_iterations_since_improvement: return record[0]

        # centroid
        x0 = [0 for _ in range(dimensions)]
        for pair in record[:-1]:
            for i, c in enumerate(pair[0]): x0[i] += c / (len(record)-1)

        # reflection
        x_reflection = x0 + alpha*(x0 - record[-1][0])
        reflection_score = f(x_reflection)
        if best <= reflection_score < record[-2][1]:
            del record[-1]
            record.append([x_reflection, reflection_score])
            continue

        # expansion
        if reflection_score < best:
            x_expansion = x0 + gamma*(x0 - record[-1][0])
            expansion_score = f(x_expansion)
            if expansion_score < reflection_score:
                del record[-1]
                record.append([x_expansion, expansion_score])
                continue
            else:
                del record[-1]
                record.append([x_reflection, reflection_score])
                continue

        # contraction
        x_contraction = x0 + rho*(x0 - record[-1][0])
        contraction_score = f(x_contraction)
        if contraction_score < record[-1][1]:
            del record[-1]
            record.append([x_contraction, contraction_score])
            continue

        # reduction
        x1 = record[0][0]
        new_record = []
        for pair in record:
            redx = x1 + sigma*(pair[0] - x1)
            score = f(redx)
            new_record.append([redx, score])
        record = new_record


if __name__ == "__main__":
    # test
    import math
    import numpy as np

    def f(x): return math.sin(x[0]) * math.cos(x[1]) * (1. / (abs(x[2]) + 1))

    print(nelder_mead(f, np.array([0., 0., 0.])))
