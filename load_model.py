
import importlib
import torch
import numpy as np

def load_model(model_path, weights_path):
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)  # Load the module
    model = model_module.Model()  # Create an instance of the model class
    model.load_state_dict(torch.load(weights_path))  # Load weights
    print("model loaded successfully")
    
    # try small data on cpu to check if the model is loaded correctly
    test_data = np.random.rand(3, 3, 128, 128)  # Example input data
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_data = test_data.to("cpu")
    model = model.to("cpu")
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        output = model.encode(test_data)
        output = model.decode(output)
    print("Model loaded successfully and output generated.")