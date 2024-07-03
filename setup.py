from setuptools import setup, find_packages

setup(
    name='deepdefend',
    version='{{VERSION_PLACEHOLDER}}',
    author='Infinitode Pty Ltd',
    author_email='infinitode.ltd@gmail.com',
    description='An open-source Python library for adversarial attacks and defenses in deep learning models.',
    long_description='An open-source Python library for adversarial attacks and defenses in deep learning models, enhancing the security and robustness of AI systems.',
    long_description_content_type='text/markdown',
    url='https://github.com/infinitode/deepdefend',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.6',
)