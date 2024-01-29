import logging

import torch
import torch._dynamo.config
from torch._dynamo import convert_frame
from torch._dynamo.eval_frame import OptimizeContext
from torch._dynamo.hooks import Hooks

torch._logging.set_logs(dynamo=logging.DEBUG)


def MyCompiler(gm, example_inputs):
    print("========================")
    gm.print_readable()
    print(example_inputs)
    print("========================")
    return gm.forward


def YouCompiler(gm, example_inputs):
    print("************************")
    gm.print_readable()
    print(example_inputs)
    print("************************")
    return gm.forward


class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x + 1
        x = x + 20
        return x


class OPT_M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = M1()

    def forward(self, x):
        return self.sub(x)


class M2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 2


class OPT_M2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = M2()

    def forward(self, x):
        x = self.sub(x)
        x = x + 100
        x = x + 1000
        return x


class MM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torch._dynamo.eval_frame import null_context, get_compiler_fn, catch_errors_wrapper

        self.m1 = OPT_M1()
        self.m2 = OPT_M2()
        torch._dynamo.config.raise_on_ctx_manager_usage = False
        hooks = Hooks(guard_export_fn=None, guard_fail_fn=None)

        mycompiler = torch._TorchCompileWrapper(MyCompiler, mode="default", options=None, dynamic=False)
        mycompiler = get_compiler_fn(mycompiler)

        self.mybackend = catch_errors_wrapper(convert_frame.convert_frame(mycompiler, hooks=hooks), hooks=hooks)
        self.mybackend_ctx_ctor = getattr(mycompiler, "backend_ctx_ctor", null_context)

        youcompiler = torch._TorchCompileWrapper(YouCompiler, mode="default", options=None, dynamic=False)
        youcompiler = get_compiler_fn(youcompiler)

        self.youbackend = catch_errors_wrapper(convert_frame.convert_frame(youcompiler, hooks=hooks), hooks=hooks)
        self.youbackend_ctx_ctor = getattr(youcompiler, "backend_ctx_ctor", null_context)

    def forward(self, x):
        with OptimizeContext(callback=self.mybackend, backend_ctx_ctor=self.mybackend_ctx_ctor):
            with OptimizeContext(callback=self.youbackend, backend_ctx_ctor=self.youbackend_ctx_ctor):
                x = self.m2(x)
            x = self.m1(x)
            return x


if __name__ == "__main__":
    m = MM()
    x = torch.ones(2)
    y = m(x)
    print(y)
