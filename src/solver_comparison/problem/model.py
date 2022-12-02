from kmerexpr.multinomial_model import multinomial_model
from kmerexpr.multinomial_simplex_model import multinomial_simplex_model

SIMPLEX = "Simplex"
SOFTMAX = "Softmax"


def unknown_model_error(name):
    raise (f"Unknown model type {name}, expected one of [{SIMPLEX}, {SOFTMAX}].")


def get_model(name: str):
    if name not in [SIMPLEX, SOFTMAX]:
        raise ValueError(unknown_model_error(name))
    if name == SIMPLEX:
        return multinomial_simplex_model
    else:
        return multinomial_model


class _FunctionGradientCache:
    def __init__(self):
        self.input_id_cache = None
        self.func_cache = None
        self.grad_cache = None

    def is_in_cache(self, theta, nograd) -> bool:
        if self.input_id_cache == id(theta):
            if nograd:
                return self.func_cache is not None
            else:
                return self.func_cache is not None and self.grad_cache is not None
        return False

    def cache(self, param, func, grad):
        self.input_id_cache = id(param)
        self.func_cache = func
        self.grad_cache = grad

    def cached_values(self, nograd):
        if nograd:
            return self.func_cache
        else:
            return self.func_cache, self.grad_cache


class Model:
    def __init__(self, model_type: str, x_file, y_file, beta=1.0, lengths=None):
        self.model_type = model_type
        self.kmerexpr_model = get_model(model_type)(
            x_file=x_file,
            y_file=y_file,
            beta=beta,
            lengths=lengths,
            solver_name=None,
        )
        self._cache = _FunctionGradientCache()

    def _logp_grad_simplex(self, param, nograd):
        if nograd:
            func = self.kmerexpr_model.logp_grad(param, nograd=nograd)
            grad = None
        else:
            func, grad = self.kmerexpr_model.logp_grad(param, nograd=nograd)
        return func, grad

    def logp_grad(self, param, nograd=False, Hessinv=False):
        if self._cache.is_in_cache(param, nograd):
            return self._cache.cached_values(nograd)

        if self.model_type == SIMPLEX:
            func, grad = self._logp_grad_simplex(param, nograd)
        elif self.model_type == SOFTMAX:
            func, grad = self.kmerexpr_model.logp_grad(param)
        else:
            raise ValueError(unknown_model_error(self.model_type))

        self._cache.cache(param, func, grad)
        return self._cache.cached_values(nograd)

    def probabilities(self, params):
        return params

    @property
    def dimension(self) -> int:
        return int(self.kmerexpr_model.T)
