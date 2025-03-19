import setuptools

# if __name__ == "__main__":
#     setuptools.setup()# Multiple top-level packages discovered in a flat-layout: ['nnunet', 'nnunetv2', 'nnUNet_results'].

if __name__ == "__main__":
    setuptools.setup(
        name="nnunetv2",
        version="2.5.1",
        packages=setuptools.find_packages(include=["nnunetv2", "nnunetv2.*"]),  # 仅安装 nnunetv2
    )
