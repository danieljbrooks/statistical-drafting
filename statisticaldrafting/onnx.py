import os

import onnx
import pandas as pd
import statisticaldrafting as sd
import torch

# Note - this code is currently untested in package format due to import issues. 
# It has been successfully tested & run in a notebook. 

def create_onnx_model(model_path, cardnames, onnx_path):
    """
    Creates an ONNX model of the network for use in browser code. 
    """
    # Creates an onnx model. 
    network = sd.DraftNet(cardnames=cardnames)
    network.load_state_dict(torch.load(model_path, weights_only=True))
    network.eval()

    # Dummy inputs (match the forward signature: collection, pack)
    dummy_collection = torch.randn(1, len(cardnames)) # [batch, num_cards]
    dummy_pack = torch.ones(1, len(cardnames)) # same shape as output for elementwise multiply

    # Save as onnx. 
    torch.onnx.export(
        network,
        (dummy_collection, dummy_pack),  # tuple for multiple inputs
        onnx_path,
        input_names=["collection", "pack"],
        output_names=["output"],
        dynamic_axes={
            "collection": {0: "batch_size"},
            "pack": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,  # >=13 for native GELU op
    )
    # PyTorch 2.x can leave a stub on disk; reload + save produces a single complete .onnx.
    onnx.save(onnx.load(onnx_path), onnx_path)
    print(f"Created {onnx_path}")

def create_all_onnx_models(model_dir="../data/models/", onnx_dir="../data/onnx/"):
    """
    Exports all models to ONNX format. 

    Runnable in notebook. 
    """
    model_names = [model_name for model_name in os.listdir(model_dir) if ".pt" in model_name]
    for model_name in model_names:

        # Get cardnames. 
        set = model_name.split("_")[0]
        pick_table = pd.read_csv(
            f"../data/cards/{set}.csv"
        )  # will be sorted.
        cardnames = pick_table["name"].tolist()
        onnx_path = onnx_dir + model_name[:-3] + ".onnx"

        try:
            create_onnx_model(model_dir + model_name, cardnames, onnx_path)
        except:
            print(f"Error for {onnx_path}")