import sys
import os
# FIXME TODO hack from ttps://stackoverflow.com/questions/6323860/sibling-package-imports/50193944#50193944
# to get tinyquant module
sys.path.insert(0, os.path.abspath('../../../../cnn/'))


from tinyquant.quantized_layer import *
from tinyquant.quantization_ema_stats import *
from tinyquant.quantization_functions import *
from tinyquant.qat_prepare import *
from tinyquant.qat_convert import *


def quantize_dynamic_torch(
        float_model:nn.Module,
        qops:set={nn.LSTM, nn.Linear},
        dtype=torch.qint8,
        ):
    # this is the torch model we want to ultimately compare against
    # and also the model we want to benchmark first before diving into the code
    float_model = float_model.to("cpu")
    quantized_model = torch.quantization.quantize_dynamic(
        float_model, qops, dtype=dtype, inplace=True
    )
    return quantized_model.to("cuda:0")


def quantize_dynamic_tinyquant(
            module: nn.Module,
            inplace: bool = True,
            qparams=None
        ) -> nn.Module:

    """ Recursively convert module with no prior QAT. """

    raise NotImplementedError()

    # module_number = module._module_number

    is_root_module = _stats is None

    if is_root_module:
        assert qparams is None
        module.__qparams__ = {"scale": float("-inf"), "zero_point": float("-inf")}
    else:
        assert type(qparams) == dict, type(qparams)
        module.__qparams__ = qparams

    if inplace == False and _stats is None:
        module = copy.deepcopy(module)

    # get dicts from top module if not passed down recursively in this function
    # _stats = module.stats if _stats is None else _stats
    # _handles = module.handles if _handles is None else _handles
    # _module_types = module.module_types if _module_types is None else _module_types
    # "need to run prep_module_qat before this which should set model.stats to dict of seen activ ranges"
    # assert isinstance(_handles, dict) and _handles, f"'_handles' argument needs to be dict of pre/post fwd hook handles of model modules and not {type(_handles)} {_handles}"

    #1. convert forward passes of all internal modules to handle only QTensors

    for name, layer in module.named_children():

        # ===== DFS down module graph ========
        convert_module(
            layer,
            inplace=True,
            qparams=module.__qparams__
        )

    # 2. convert known layer types and remove forward hooks on a basis of spaghetti


    # ################  end remove pre / post hooks ###################


    if isinstance(module, nn.Conv2d) \
        or isinstance(module, nn.Linear) \
        or isinstance(module, QuantizableResConnection) \
        or isinstance(module, nn.BatchNorm2d):


        module.forward = _dynamic_convert_layer_forward_impl(
           module
        )
        # also convert identity connection for residual connection
        if isinstance(module, QuantizableResConnection):
            module.add = _dynamic_convert_quantized_add_impl(
                    module
            )
            module.is_quantized = True

    elif is_activation:

        # case by case implementation:
        if isinstance(module, nn.ReLU6):
            module.forward = _dynamic_convert_relu6_layer_forward_impl(
                module,
            )
        elif isinstance(module, nn.ReLU):
            module.forward = _dynamic_convert_relu_layer_forward_impl(
                module
            )
        elif isinstance(module, nn.Identity):
            pass
        else:
            print(f"Dont yet know how to implement quantized forward pass of {type(module)} (Or handled this elsewhere (QuantizableResConnection)")

    elif isinstance(module, ConvBNnofold):
        _dynamic_convert_bnnofold_layer_forward(
            module
        )

    elif isinstance(module, ConvBNfoldable):
        # stop simulating Batchnom fwd passes
        # module.simulate_folding_params_handle.remove()
        # module.qat_convert_by_removing_me.remove()

        # fold weights!
        _dynamic_convert_convbnfoldable_layer_forward(
            module
        )

    # 3. convert forward pass of module to dequantize at end
    if is_root_module:
        # search for tinyquant.quantization_functions.Quantization objects for the input inside of the prepared model modules
        pass


    return module



