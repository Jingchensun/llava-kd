
## 🛠️ Installation

- Based on python3.12 and torch-2.6.0

- Prepare the environment
  ```shell
  python3.12 -m pip install --no-cache-dir --upgrade -r requirements.txt
  python3.12 -m pip install numpy==1.26.2
  python3.12 -m pip install urllib3==1.26.6
  ```

- Install cuda12.6
  ```shell
  sh cuda_12.9.1_575.57.08_linux.run
  ```

- Install Cusparselt
  ```shell
  cd ../LLaVA_KD_whls/
  rpm -i cusparselt-local-repo-rhel9-0.7.1-1.0-1.x86_64.rpm
  dnf clean all
  dnf -y install libcusparselt0 libcusparselt-devel
  ```

- Install bitsandbytes
  ```shell
  cd ../LLaVA_KD_whls/bitsandbytes-0.46.0
  python3.12 setup.py install 
  ```

- Install deepspeed
  ```shell
  python3.12 -m pip install ptflops
  python3.12 -m pip install deepspeed==0.14.4


## LLaVA-KD Weights
| Model                                                   | Vision Encoder                                               | LLM                                                           |  CKPTs
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------  | ------------------------------------------------------------  |
| LLaVA-KD-1B-Base-Qwen1.5 | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) |  [Qwen/Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B)     | [LLaVA-KD-Base-siglip-Qwen1.5-0.5B]()
| LLaVA-KD-2B-Base-Qwen1.5 | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) |  [Qwen/Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B)     | [LLaVA-KD-Base-siglip-Qwen1.5-1.8B]()
| LLaVA-KD-1B-Base-Qwen2.5 | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) |  [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)     | [LLaVA-KD-Base-siglip-Qwen2.5-0.5B]()
| LLaVA-KD-2B-Base-Qwen2.5 | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) |  [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)     | [LLaVA-KD-Base-siglip-Qwen2.5-1.5B]()

## :computer: Evaluation
Please evaluate the model according to [Evaluation.md](docs/Evaluation.md).

## Quickstart
Download the Pre-trained VisualEnc, LLM, LLaVAKD weights to the `./pretrained_ckpt`. And then:
  ```shell
  python quick_inference.py --model_path ./pretrained_ckpt/LLaVAKD_Model_Path --image_file ./image_test/img_test_1.jpg  --query "What is that orange thing behind the girl?"
  ```
<p align="center">
  <img src="assets/Visual.jpg" alt="accuracy" width="100%">
</p>


## 💘 Acknowledgements
We thank the great works [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory), [LLaVA](https://github.com/haotian-liu/LLaVA) for providing assistance for our research.

