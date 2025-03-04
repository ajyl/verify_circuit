"""
Utility functions for grabbing the residual stream.
"""

from typing import Any, Generator, Sequence, cast, Callable, Iterable, Literal
from collections import defaultdict
from contextlib import contextmanager
import torch
from torch import Tensor, nn


def untuple_tensor(x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


def get_module(model: nn.Module, name: str) -> nn.Module:
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def _create_read_hook(layer_name: str, records: dict[str, list[Tensor]]) -> Any:
    """Create a hook function that records the model activation at :layer_name:"""

    def hook_fn(_module: Any, _inputs: Any, _outputs: Any) -> Any:
        # _inputs[0]: [batch, seq, d_model]
        # _outputs[0]: [batch, seq, d_model],
        activation = untuple_tensor(_outputs)
        if not isinstance(cast(Any, activation), Tensor):
            raise ValueError(
                f"Expected a Tensor reading model activations, got {type(activation)}"
            )

        _activation = activation.clone().detach()
        records[layer_name].append(_activation)
        return _outputs

    return hook_fn


@contextmanager
def record_activations(
    model: nn.Module,
    module_names: list[str],
) -> Generator[dict[str, list[Tensor]], None, None]:
    """
    Record the model activations at each layer of type `layer_type`.
    This function will record every forward pass through the model
    at all layers of the given layer_type.

    Args:
        model: The model to record activations from.
        modules: List of modules to grab activations from.
    Example:
    """
    recorded_activations: dict[int, list[Tensor]] = defaultdict(list)
    hooks = []

    for _module_name in module_names:
        module = get_module(model, _module_name)

        # hook_fn: hook(module, input, output)
        hook_fn = _create_read_hook(
            _module_name,
            recorded_activations,
        )
        handle = module.register_forward_hook(hook_fn)
        hooks.append(handle)

    try:
        yield recorded_activations
    finally:
        for hook in hooks:
            hook.remove()
