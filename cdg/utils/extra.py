
def arrange_as_matrix(X):
    return X.reshape(1, X.shape[0]) if X.ndim ==1 else X

def get_real_eig(A, tol=1e-5):
    """
    Compute the real eigenvalues of a symmetric matrix.

    :param A: (n, n) symmetric np.array
    :param tol: tolerance in considering a complex number as a real
    :return:
        - eigenvalues : sorted in descending order
        - eigenvectors : corresponding to eigenvalues
    """
    
    import numpy as np
    import cdg.utils.errors
    
    # check symmetry
    if not np.allclose(A, A.T, atol=tol):
        raise NotImplementedError("Matrix A needs to be symmetric")

    # compute spectrum
    # https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
    # https://stackoverflow.com/a/8765592
    val_complex, vec_complex = np.linalg.eigh(A)
    # cast to real
    val_real = np.real(val_complex)
    vec_real = np.real(vec_complex)
    if not np.allclose(val_real, val_complex, atol=tol) \
            or not np.allclose(vec_real, vec_complex, atol=tol):
        raise cdg.utils.errors.CDGImpossible('complex numbers')

    # sort output
    sorted_index = np.argsort(val_real)[::-1]
    eigenvalues = val_real[sorted_index]
    eigenvectors = vec_real[:, sorted_index]

    return eigenvalues, eigenvectors

def anneal(no_annealing, fun, verbose=False, desc='annealing'):
    from tqdm import tqdm
    val = None
    for _ in tqdm(range(no_annealing), disable=not verbose, desc=desc):
        prot_id_tmp, val_tmp = fun()
        if val is None or val_tmp < val:
            prot_id = prot_id_tmp.copy()
            val = val_tmp
    return prot_id.copy(), val

def _preprocess_exp(exp):
    possible_exp = [("Letter", 'd.Let'),
                    ("Mutagenicity", 'd.Mut'),
                    ("AIDS", 'd.AIDS'),
                    ("Markov", 'd.Mk'),
                    ("Delaunay", 'd.Del'),
                    ("Dog", 'd.Dog'),
                    ("Human", 'd.Hum'),
                    ("SBM", 'd.SBM')]

    for p in possible_exp:
        if p[0] in exp:
            return p[1]
    return exp

def _postprocess_latex(latex_dict):
    text = latex_dict['str_format'].format(open=latex_dict['open'],
                                           header=latex_dict['header'],
                                           body=latex_dict['body'],
                                           close=latex_dict['close'])
    return text.replace('_', '-')

def gather_results(args_to_parse, setting_list, postprocessing_fun=_postprocess_latex):
    import os.path
    import argparse
    from cdg.utils import logger
    
    # Parse args
    help_menu = 'List of figures of merit available:\n'
    from cdg.simulation import read_resultfile, get_figures_merit
    for k, v in get_figures_merit().items():
        help_menu += '\t\t{}: {}\n'.format(str(k), str(v))

    parser = argparse.ArgumentParser(description=help_menu)
    parser.add_argument('-f', '--filename', type=str,
                        required=True,
                        help='file containing the list of zipped experiment results')
    parser.add_argument('-l', '--latex', type=str,
                        default='dca',
                        help='format of the LaTeX table entry' + help_menu)

    args = parser.parse_args(args_to_parse)

    # Read the list of zipped result file
    assert os.path.isfile(args.filename), '{}\n{}'.format(args.filename, parser.print_help())
    logger.info("reading %s ..." % args.filename)
    with open(args.filename) as f:
        list_exp = [line.strip() for line in f if len(line)>1]
        
    # Arrange results in a dictionary
    logger.info('retrieving results...')
    experiments=[]
    for exp in list_exp:
        # read experiment result file
        logger.info('reading experiment {}'.format(exp))
        results, settings = read_resultfile(zipped_file_path=exp, figure_merit=args.latex, params=setting_list)
        experiments.append({'experiment': exp, 'results': results, 'settings': settings})
    # sort
    experiments_sorted = sorted(experiments, key=lambda e: (e['settings']['dataset'], e['settings']['cpm']))

    # Create latex table
    open_latex = '\\begin{table}\n\\begin{tabular}{c'
    header = ''
    content = ''
    first_run = True
    close_latex = '\\end{tabular}\n\\end{table}'
    for e in experiments_sorted:
        # experiment name
        if first_run:
            header += '{:12}'.format('prep_exp_name')
        content += '{:12}'.format(str(_preprocess_exp(e['experiment'])))
        # parameter setting
        for s in settings.keys():
            content += ' & {:12}'.format(str(e['settings'][s]))
            if first_run:
                header += ' & {:12}'.format(s)
                open_latex += 'c'
        if first_run:
            open_latex += '|'
        # results
        for r in results.keys():
            content += ' & {:6}'.format(str(e['results'][r]))
            if first_run:
                header += ' & {:6}'.format(r)
                open_latex += 'c'
        # closing
        content += '\\\\\n'
        if first_run:
            # header = header[:-2]
            open_latex += '}'
            first_run = False
    #assemble table
    latex_dict = {'str_format': '\n{open}\n{header}\\\\\n\\hline\n{body}\\hline\n{close}\n',
                  'open': open_latex,
                  'header': header,
                  'body': content,
                  'close': close_latex}
    # latex_table = '\n{}\n{}\\\\\n\\hline\n{}\\hline\n{}\n'.format(open_latex, header, content, close_latex)
    latex_table = postprocessing_fun(latex_dict)
    logger.info(latex_table)
    return latex_table

def permutation_test(x, test_fun, observed, repetitions, n_jobs=1, is_matrix=False, verbose=False):
    """
    Functionality to perform a permutation test on test_fun.
    :param x: input data
    :param test_fun: (stat = fun(x)) function computing the statistics over sample x
    :param observed: (float) observed statistics to compare with
    :param repetitions: (int) number of permutations to perform
    :param is_matrix: (bool, def=False) whether input x has to be permuted only row-wise or also column-wise
    :param verbose:
    :return: estimated p-value
    """
    assert repetitions is not None

    tqdm_disable = not verbose
    T = x.shape[0]
    
    from tqdm import tqdm
    import numpy as np
    import joblib
    
    # for _ in tqdm(range(repetitions), disable=tqdm_disable, desc='permutation test'):
    def parallel_fun(xf):
        perm = np.random.permutation(T).reshape(-1, 1)
        if is_matrix:
            stat = test_fun(x=xf[perm, perm.T])
        else:
            stat = test_fun(x=xf[perm])
        return 1 if stat >= observed else 0


    # for _ in tqdm(range(repetitions), disable=tqdm_disable, desc='permutation test'):
    def parallel_fun_for(xf, nit):
        ct = 0
        it = 0
        for _ in range(nit):
            perm = np.random.permutation(T).reshape(-1, 1)
            if is_matrix:
                stat = test_fun(x=xf[perm, perm.T])
            else:
                stat = test_fun(x=xf[perm])
            if stat >= observed:
                ct += 1
            it += 1
        return ct, it
    # ct = 0
    # for _ in tqdm(range(repetitions), disable=tqdm_disable, desc="permutation"):
    #     ct += parallel_fun(xf=x)

    num_geq_observed = 0
    if repetitions > 0:
        step = repetitions//n_jobs if n_jobs > 0 else 20
        ll = [i*step for i in range(repetitions//step)] + [repetitions]
        output = joblib.Parallel(n_jobs=n_jobs, verbose=verbose) \
                (joblib.delayed(parallel_fun_for)(xf=x, nit=ll[i+1]-ll[i]) for i in range(len(ll)-1))
        num_geq_observed = sum([o[0] for o in output])
        assert sum([o[1] for o in output]) == repetitions
    return (num_geq_observed + 1) / (repetitions + 1)
