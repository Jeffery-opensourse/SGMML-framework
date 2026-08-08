"""SGMML: Semantics-Guided Multimodal Machine Learning.

A framework for predicting the flexural strength of C/C-SiC composites by
combining a LoRA-fine-tuned LLM text encoder with structured numerical
features, as described in:

    A Semantics-Guided Multimodal Machine Learning Framework for Predicting
    the Flexural Strength of C/C-SiC Composites.

Light-weight modules (data, metrics, fusion, regressor) are imported
directly. The heavy modules (text_encoder, llm_api, trainer) are exposed
through ``__getattr__`` so that importing the package never requires
torch, transformers or an OpenAI client unless those parts are used.
"""
from . import config, data, metrics

__version__ = "1.0.0"

_LAZY = ("fusion", "regressor", "ml_models", "text_encoder", "llm_api", "trainer")


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY))


__all__ = ["config", "data", "metrics"] + list(_LAZY)
