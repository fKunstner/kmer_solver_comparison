from kmerexpr.multinomial_model import multinomial_model
from kmerexpr.multinomial_simplex_model import multinomial_simplex_model

SIMPLEX = "Simplex"
SOFTMAX = "Softmax"


def error_unknown(name):
    raise ValueError(
        f"Unknown model type {name}, expected one of [{SIMPLEX}, {SOFTMAX}]."
    )


def get_model(name: str):
    if name not in [SIMPLEX, SOFTMAX]:
        error_unknown(name)
    if name == SIMPLEX:
        return multinomial_simplex_model
    else:
        return multinomial_model


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

    def logp_grad(self, theta=None, nograd=False, Hessinv=False):
        func, grad = None, None
        if self.model_type == SIMPLEX:
            if nograd:
                func = self.kmerexpr_model.logp_grad(theta, nograd=nograd)
            else:
                func, grad = self.kmerexpr_model.logp_grad(theta, nograd=nograd)
        elif self.model_type == SOFTMAX:
            func, grad = self.kmerexpr_model.logp_grad(theta)
        else:
            error_unknown(self.model_type)

        if nograd:
            return func
        else:
            return func, grad

    def probabilities(self, params):
        return params

    @property
    def dimension(self) -> int:
        return int(self.kmerexpr_model.T)
