from setuptools import setup, find_packages

setup(
    name="nerf-studio-backend",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "websockets>=12.0",
        "torch>=2.6.0",
        "torchvision>=0.17.0",
        "numpy>=1.24.3",
        "Pillow>=10.0.1",
        "opencv-python>=4.8.1.78",
        "trimesh>=4.0.5",
        "pyglet>=2.0.9",
        "psutil>=5.9.5",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
        ]
    }
) 