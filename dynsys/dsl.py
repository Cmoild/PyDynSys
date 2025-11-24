from lark import Lark, Transformer, v_args

GRAMMAR = r"""
    start: stmt+
    stmt: deriv_assign

    deriv_assign: NAME "'" "=" expr   -> deriv

    ?expr: expr "+" term   -> add
         | expr "-" term   -> sub
         | term

    ?term: term "*" pow    -> mul
         | term "/" pow    -> div
         | pow

    ?pow: atom "^" pow     -> pow
        | atom

    SIGN: "+" | "-"

    ?atom: SIGN atom -> unary
         | NUMBER          -> number
         | NAME "(" args? ")" -> func
         | NAME             -> var
         | "(" expr ")"

    args: expr ("," expr)*

    %import common.CNAME -> NAME
    %import common.NUMBER -> NUMBER
    %import common.WS
    %ignore WS
"""

MATH_FUNC_MAP = {
    "sin": "sinf",
    "cos": "cosf",
    "tan": "tanf",
    "exp": "expf",
    "log": "logf",
    "sqrt": "sqrtf",
    "pow": "powf",
    # extend as needed
}


@v_args(inline=True)
class ToC(Transformer[list[tuple[str, str]]]):
    def __init__(self, const_names: list[str]):
        super().__init__()
        self.const_index = {name: i for i, name in enumerate(const_names)}
        self.var_index: dict[str, int] = {}
        self.var_order: list[str] = []

    # convert the top-level start node into a plain list of statements
    def start(self, *stmts):
        # each stmt should already be a (varname, expr_str) tuple
        return list(stmts)

    # unwrap stmt produced by grammar (stmt -> deriv_assign)
    def stmt(self, item):
        return item

    def _make_var(self, name: str) -> str:
        """Record variable name and return its X[...] reference string."""
        if name not in self.var_index:
            self.var_index[name] = len(self.var_order)
            self.var_order.append(name)
        return f"X[{self.var_index[name]}]"

    # deriv: NAME "'" "=" expr
    def deriv(self, name, expr) -> tuple[str, str]:
        varname = str(name)
        # ensure lhs variable is registered (even if it doesn't appear in RHS)
        self._make_var(varname)
        expr_str = expr
        return (varname, expr_str)

    # binary ops
    def add(self, a, b):
        return f"({a} + {b})"

    def sub(self, a, b):
        return f"({a} - {b})"

    def mul(self, a, b):
        return f"({a} * {b})"

    def div(self, a, b):
        return f"({a} / {b})"

    def pow(self, a, b):
        return f"powf({a}, {b})"

    def unary(self, sign, value):
        # sign — Token('-', '-') или Token('+', '+')
        s = str(sign)
        if s == "-":
            return f"(-{value})"
        else:
            # unary plus
            return f"({value})"

    def number(self, token):
        s = str(token)
        if "." in s or "e" in s or "E" in s:
            return f"{s}f"
        else:
            return f"{s}.0f"

    def var(self, name_token):
        name = str(name_token)
        if name in self.const_index:
            return f"C[{self.const_index[name]}]"
        else:
            return self._make_var(name)

    def func(self, name_token, *args):
        fname = str(name_token)
        mapped = MATH_FUNC_MAP.get(fname, fname)
        args_s = ", ".join(args) if args else ""
        return f"{mapped}({args_s})"


def compile_dsl_to_c(
    dsl_code: str, constants: dict[str, float], func_name: str = "generated_func"
) -> tuple[str, list[str], list[str]]:
    const_order = list(constants.keys())
    parser = Lark(
        GRAMMAR, parser="lalr", propagate_positions=False, maybe_placeholders=False
    )
    transformer = ToC(const_order)

    tree = parser.parse(dsl_code)
    derivs = transformer.transform(tree)  # now a list of (varname, expr_str)

    # If for some reason transform returns a Tree, be defensive:
    if hasattr(derivs, "children"):
        derivs = derivs.children

    lines = []
    for item in derivs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise RuntimeError(
                "Internal: expected (varname, expr) tuple from transformer"
            )
        varname, expr_str = item
        idx = transformer.var_index[varname]
        lines.append(f"    X_dot[{idx}] = {expr_str};")

    body = "\n".join(lines)
    c_code = f"""void {func_name}(const float* X, const float* C, float* X_dot) {{
{body}
}}"""
    return c_code, transformer.var_order, const_order
