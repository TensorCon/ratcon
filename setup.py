from setuptools import setup

setup(
    name='order',
    version='0.0.1',
    description='optimize tensor network contractions',
    url='https://github.com/TensorCon/order',
    author_email='aneksteind@gmail.com',
    license='GPLv3',
    packages=[
        'opt/gencon'
    ],
    install_requires=[
        'networkx>=2.3',
        'click==7.0',
        'deap>=1.2.2',
        'pytest==5.0.1',
        'oct2py==4.0.6',
        'numpy==1.17.0',
        'matplotlib==2.2.2;implementation_name != "pypy"',
        'scipy>=1.3.1;implementation_name != "pypy"'
    ],
    python_requires='>=3.6.1',
    entry_points = {
        'console_scripts': [
            'ratcon=opt.run:ratcon',
            'netcon=opt.run:netcon',
            'gencon=opt.run:gencon',
            'test-gen=opt.run:generate_tests'
        ]
    }
)