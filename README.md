
<div align="center">
  <img src="images/cloud.png" alt="Algorithm icon">
  <h1 align="center">infer_google_vision_image_properties</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_google_vision_image_properties">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_google_vision_image_properties">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_google_vision_image_properties/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_google_vision_image_properties.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>


Face detection using Google cloud vision API. 

**Running this algorithm requires**: 
- **a Google Cloud Vision API Key**
- **a Google Cloud account with Cloud Vision API enable**

**Pease refer to the 'Advanced Usage' section for guidance on how to set these up.**


![colormap bird](https://raw.githubusercontent.com/Ikomia-hub/infer_google_vision_image_properties/main/images/output.png)


## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_google_vision_image_properties", auto_connect=True)

# Set parameters
algo.set_parameters({
    'google_application_credentials':'PATH/TO/YOUR/GOOGLE/CLOUD/VISION/API/KEY.json '
})

# Run on your image
wf.run_on(url='https://images.unsplash.com/photo-1501179691627-eeaa65ea017c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2670&q=80')

```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).



## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_google_vision_image_properties", auto_connect=True)

# Set parameters
algo.set_parameters({
    'google_application_credentials':'PATH/TO/YOUR/GOOGLE/CLOUD/VISION/API/KEY.json '
})

# Run on your image
wf.run_on(url='https://images.unsplash.com/photo-1501179691627-eeaa65ea017c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2670&q=80')

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

## :fast_forward: Advanced usage 

 ### :bulb: How to generate a Google Cloud Vision API Key and enable Cloud Vision API?
- [YT video tutorial](https://www.youtube.com/watch?v=kZ3OL3AN_IA&t=157s)
- [Blog tutorial](https://daminion.net/docs/how-to-get-google-cloud-vision-api-key/)


### :key: Set the Google Cloud Vision API Key in your environment variable. 
[Permanently setting the API Key in your environment variable](https://medium.com/@kapilgorve/set-environment-variable-in-windows-and-wsl-linux-in-terminal-c5e11138e807) enables the use of this algorithm without having to define the 'google_application_credentials' parameter every time.


*Note: the key will be require for deployments.*



###  :red_circle: Deployment Limitations
This algorithm necessitates authentication to Google Cloud services via API keys. Consequently, it will not operate offline (e.g., in AWS Lambda) or in environments without internet access to communicate with Google Cloud services.

Ensure to manage API keys securely and avoid exposing them in public repositories or forums to prevent unauthorized usage. If the keys are compromised, be sure to revoke them immediately and generate new keys in the Google Cloud Console.

